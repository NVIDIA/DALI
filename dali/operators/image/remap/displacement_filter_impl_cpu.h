// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_OPERATORS_IMAGE_REMAP_DISPLACEMENT_FILTER_IMPL_CPU_H_
#define DALI_OPERATORS_IMAGE_REMAP_DISPLACEMENT_FILTER_IMPL_CPU_H_

#include <array>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/operators/image/remap/displacement_filter.h"
#include "dali/pipeline/data/views.h"
#include "dali/kernels/kernel_params.h"
#include "dali/kernels/imgproc/sampler.h"
#include "dali/core/convert.h"
#include "dali/core/static_switch.h"

namespace dali {

template <DALIInterpType interp_type, bool per_channel,
          typename Out, typename In, typename Displacement, typename Border>
void Warp(
    const kernels::OutTensorCPU<Out, 3> &out,
    const kernels::InTensorCPU<In, 3> &in,
    Displacement &displacement,
    Border border) {
  DALI_ENFORCE(in.shape[2] == out.shape[2], "Number of channels in input and output must match");
  int outH = out.shape[0];
  int outW = out.shape[1];
  int C = out.shape[2];
  int inH = in.shape[0];
  int inW = in.shape[1];

  kernels::Sampler2D<interp_type, In> sampler(kernels::as_surface_HWC(in));

  for (int y = 0; y < outH; y++) {
    Out *out_row = out(y, 0);
    for (int x = 0; x < outW; x++) {
      if (per_channel) {
        for (int c = 0; c < C; c++) {
          auto p = displacement(y, x, c, inH, inW, C);
          sampler(&out_row[C*x], p, c, border);
        }
      } else {
        auto p = displacement(y, x, 0, inH, inW, C);
        sampler(&out_row[C*x], p, border);
      }
    }
  }
}

template <class Displacement, bool per_channel_transform>
class DisplacementFilter<CPUBackend, Displacement, per_channel_transform>
    : public Operator<CPUBackend> {
 public:
  explicit DisplacementFilter(const OpSpec &spec)
      : Operator(spec),
        displace_(num_threads_, Displacement(spec)),
        interp_type_(spec.GetArgument<DALIInterpType>("interp_type")) {
    has_mask_ = spec.HasTensorArgument("mask");
    DALI_ENFORCE(
        interp_type_ == DALI_INTERP_NN || interp_type_ == DALI_INTERP_LINEAR,
        "Unsupported interpolation type, only NN and LINEAR are supported for "
        "this operation");
    if (!spec.TryGetArgument<float>(fill_value_, "fill_value")) {
      int int_value = 0;
      if (!spec.TryGetArgument<int>(int_value, "fill_value")) {
        DALI_FAIL("Invalid type of argument \"fill_value\". Expected int or float");
      }
      fill_value_ = int_value;
    }
  }

  ~DisplacementFilter() override {
    for (auto &d : displace_) {
      d.Cleanup();
    }
  }

  template <typename Out, typename In, DALIInterpType interp>
  void RunWarp(Tensor<CPUBackend> &output, const Tensor<CPUBackend> &input, int thread_idx) {
    auto &displace = displace_[thread_idx];
    In fill[1024];
    auto in = view_as_tensor<const Out, 3>(input);
    auto out = view_as_tensor<In, 3>(output);

    for (int i = 0; i < in.shape[2]; i++) {
      fill[i] = fill_value_;
    }

    Warp<interp, per_channel_transform>(out, in, displace, fill);
  }

  void RunSample(HostWorkspace &ws, int sample_idx, int thread_idx) {
    const auto &input = ws.InputRef<CPUBackend>(0);
    auto &output = ws.OutputRef<CPUBackend>(0);

    PrepareDisplacement(ws, sample_idx, thread_idx);

    if (!has_mask_ || (*mask_)[sample_idx].data<int>()[0]) {
      const auto &in_tensor = input[sample_idx];
      auto &out_tensor = output[sample_idx];

      switch (interp_type_) {
        case DALI_INTERP_NN:
          if (IsType<float>(input.type())) {
            RunWarp<float, float, DALI_INTERP_NN>(out_tensor, in_tensor, thread_idx);
          } else if (IsType<uint8_t>(input.type())) {
            RunWarp<uint8_t, uint8_t, DALI_INTERP_NN>(out_tensor, in_tensor, thread_idx);
          } else {
            DALI_FAIL("Unexpected input type " + input.type().name());
          }
          break;
        case DALI_INTERP_LINEAR:
          if (IsType<float>(input.type())) {
            RunWarp<float, float, DALI_INTERP_LINEAR>(out_tensor, in_tensor, thread_idx);
          } else if (IsType<uint8_t>(input.type())) {
            RunWarp<uint8_t, uint8_t, DALI_INTERP_LINEAR>(out_tensor, in_tensor, thread_idx);
          } else {
            DALI_FAIL("Unexpected input type " + input.type().name());
          }
          break;
        default:
          DALI_FAIL(
              "Unsupported interpolation type,"
              " only NN and LINEAR are supported for this operation");
      }
    } else {
      const auto &in_tensor = input[sample_idx];
      auto &out_tensor = output[sample_idx];
      out_tensor.Copy(in_tensor, 0);
    }
  }

  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const HostWorkspace &ws) override {
    const auto &input = ws.InputRef<CPUBackend>(0);
    output_desc.resize(1);
    output_desc[0].shape = input.shape();
    output_desc[0].type = input.type();
    return true;
  }

  void RunImpl(HostWorkspace &ws) override {
    const auto &input = ws.InputRef<CPUBackend>(0);
    const auto &shape = input.shape();
    auto &output = ws.OutputRef<CPUBackend>(0);
    output.SetLayout(input.GetLayout());

    if (has_mask_) {
      mask_ = &(ws.ArgumentInput("mask"));
    }

    auto &tp = ws.GetThreadPool();

    for (int sample_idx = 0; sample_idx < shape.num_samples(); sample_idx++) {
      tp.AddWork([&, sample_idx](int thread_idx) { RunSample(ws, sample_idx, thread_idx); },
                 shape.tensor_size(sample_idx));
    }

    tp.RunAll();
  }

  template <typename U = Displacement>
  std::enable_if_t<HasParam<U>::value> PrepareDisplacement(HostWorkspace &ws, int sample_idx,
                                                           int thread_idx) {
    auto *p = &displace_[thread_idx].param;
    displace_[thread_idx].Prepare(p, spec_, ws, sample_idx);
  }

  template <typename U = Displacement>
  std::enable_if_t<!HasParam<U>::value> PrepareDisplacement(HostWorkspace &, int, int) {}

  USE_OPERATOR_MEMBERS();
  using Operator<CPUBackend>::RunImpl;

 private:
  std::vector<Displacement> displace_;
  DALIInterpType interp_type_;
  float fill_value_;

  bool has_mask_;
  const TensorVector<CPUBackend> *mask_ = nullptr;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_REMAP_DISPLACEMENT_FILTER_IMPL_CPU_H_
