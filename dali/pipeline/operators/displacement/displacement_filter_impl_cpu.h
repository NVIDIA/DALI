// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_DISPLACEMENT_DISPLACEMENT_FILTER_IMPL_CPU_H_
#define DALI_PIPELINE_OPERATORS_DISPLACEMENT_DISPLACEMENT_FILTER_IMPL_CPU_H_

#include <array>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/pipeline/operators/displacement/displacement_filter.h"
#include "dali/pipeline/data/views.h"
#include "dali/kernels/kernel_params.h"
#include "dali/kernels/common/convert.h"
#include "dali/kernels/static_switch.h"

namespace dali {

template <DALIInterpType interp>
struct Interpolation;

template <>
struct Interpolation<DALI_INTERP_NN> {

  template <typename In>
  static const In *value_at(const kernels::InTensorCPU<In, 3> &input, int x, int y, const In *border_value) {
    if (x < 0 || x >= input.shape[1] ||
        y < 0 || y >= input.shape[0]) {
      return border_value;
    } else {
      return input(y, x);
    }
  }

  struct Sample {
    Sample() = default;
    Sample(int x, int y) : x(x), y(y) {}
    int x, y;

    template <typename T, typename In>
    void operator()(T *pixel, const kernels::InTensorCPU<In, 3> &input, const In *border_value) const {
      auto *v = value_at(input, x, y, border_value);
      for (int c = 0; c < input.shape[2]; c++) {
        pixel[c] = kernels::clamp<T>(v[c]);
      }
    }

    template <typename T, typename In>
    void operator()(T *pixel, const kernels::InTensorCPU<In, 3> &input, const In *border_value, int channel) const {
      auto *v = value_at(input, x, y, border_value);
      pixel[channel] = kernels::clamp<T>(v[channel]);
    }
  };

  Sample operator()(float x, float y) const {
    return {
      static_cast<int>(floorf(x)),
      static_cast<int>(floorf(y))
    };
  }
};


template <>
struct Interpolation<DALI_INTERP_LINEAR> {
  struct Sample {
    float x, y;

    template <typename T, typename In>
    void operator()(T *pixel, const kernels::InTensorCPU<In, 3> &input, const In *border_value) const {
      using NN = Interpolation<DALI_INTERP_NN>;
      int x0 = floorf(x);
      int y0 = floorf(y);
      const In *s00 = NN::value_at(input, x0,   y0,   border_value);
      const In *s01 = NN::value_at(input, x0+1, y0,   border_value);
      const In *s10 = NN::value_at(input, x0,   y0+1, border_value);
      const In *s11 = NN::value_at(input, x0+1, y0+1, border_value);
      float qx = x - x0;
      float px = 1 - qx;
      float qy = y - y0;
      for (int c = 0; c < input.shape[2]; c++) {
        float s0 = s00[c] * px + s01[c] * qx;
        float s1 = s10[c] * px + s11[c] * qx;
        pixel[c] = kernels::clamp<T>(s0 + (s1 - s0) * qy);
      }
    }

    template <typename T, typename In>
    void operator()(T *pixel, const kernels::InTensorCPU<In, 3> &input, const In *border_value, int channel) const {
      using NN = Interpolation<DALI_INTERP_NN>;
      int x0 = floorf(x);
      int y0 = floorf(y);
      const In *s00 = NN::value_at(input, x0,   y0,   border_value);
      const In *s01 = NN::value_at(input, x0+1, y0,   border_value);
      const In *s10 = NN::value_at(input, x0,   y0+1, border_value);
      const In *s11 = NN::value_at(input, x0+1, y0+1, border_value);
      float qx = x - x0;
      float px = 1 - qx;
      float qy = y - y0;
      float s0 = s00[channel] * px + s01[channel] * qx;
      float s1 = s10[channel] * px + s11[channel] * qx;
      pixel[channel] = kernels::clamp<T>(s0 + (s1 - s0) * qy);
    }
  };

  Sample operator()(float x, float y) const {
    return { x, y };
  }
};

template <DALIInterpType interp_type, bool per_channel,
          typename Out, typename In, typename Displacement>
void Warp(
    const kernels::OutTensorCPU<Out, 3> &out,
    const kernels::InTensorCPU<In, 3> &in,
    Displacement &displacement,
    const Out *fillValue) {
  DALI_ENFORCE(in.shape[2] == out.shape[2], "Number of channels in input and output must match");
  int outH = out.shape[0];
  int outW = out.shape[1];
  int C = out.shape[2];
  int inH = in.shape[0];
  int inW = in.shape[1];

  Interpolation<interp_type> interp;

  for (int y = 0; y < outH; y++) {
    Out *out_row = out(y, 0);
    for (int x = 0; x < outW; x++) {
      if (per_channel) {
        for (int c = 0; c < C; c++) {
          auto p = displacement.template operator()<float>(y, x, c, inH, inW, C);
          interp(p.x, p.y)(&out_row[C*x], in, fillValue, c);
        }
      } else {
        auto p = displacement.template operator()<float>(y, x, 0, inH, inW, C);
        interp(p.x, p.y)(&out_row[C*x], in, fillValue);
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
  void RunWarp(SampleWorkspace *ws, int idx) {
    auto &input = ws->Input<CPUBackend>(idx);
    auto &output = ws->Output<CPUBackend>(idx);

    auto &displace = displace_[ws->thread_idx()];
    In fill[1024];
    auto in = view_as_tensor<const Out, 3>(input);
    auto out = view_as_tensor<In, 3>(output);

    for (int i = 0; i < in.shape[2]; i++) {
      fill[i] = fill_value_;
    }

    Warp<interp, per_channel_transform>(out, in, displace, fill);

  }

  void RunImpl(SampleWorkspace *ws, const int idx) override {
    DataDependentSetup(ws, idx);

    auto &input = ws->Input<CPUBackend>(idx);

    if (!has_mask_ || mask_->template data<bool>()[ws->data_idx()]) {
      switch (interp_type_) {
        case DALI_INTERP_NN:
          if (IsType<float>(input.type())) {
            RunWarp<float, float, DALI_INTERP_NN>(ws, idx);
          } else if (IsType<uint8_t>(input.type())) {
            RunWarp<uint8_t, uint8_t, DALI_INTERP_NN>(ws, idx);
          } else {
            DALI_FAIL("Unexpected input type " + input.type().name());
          }
          break;
        case DALI_INTERP_LINEAR:
          if (IsType<float>(input.type())) {
            RunWarp<float, float, DALI_INTERP_LINEAR>(ws, idx);
          } else if (IsType<uint8_t>(input.type())) {
            RunWarp<uint8_t, uint8_t, DALI_INTERP_LINEAR>(ws, idx);
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
      auto &output = ws->Output<CPUBackend>(idx);
      output.Copy(input, ws->stream());
    }
  }

  template <typename U = Displacement>
  typename std::enable_if<HasParam<U>::value>::type PrepareDisplacement(
      SampleWorkspace *ws) {
    auto *p = &displace_[ws->thread_idx()].param;
    displace_[ws->thread_idx()].Prepare(p, spec_, ws, ws->data_idx());
  }

  template <typename U = Displacement>
  typename std::enable_if<!HasParam<U>::value>::type PrepareDisplacement(
      SampleWorkspace *) {}

  /**
   * @brief Do basic input checking and output setup
   * assuming output_shape = input_shape
   */
  virtual void DataDependentSetup(SampleWorkspace *ws, const int idx) {
    auto &input = ws->Input<CPUBackend>(idx);
    auto &output = ws->Output<CPUBackend>(idx);
    output.ResizeLike(input);
  }

  void SetupSharedSampleParams(SampleWorkspace *ws) override {
    if (has_mask_) {
      mask_ = &(ws->ArgumentInput("mask"));
    }
    PrepareDisplacement(ws);
  }

  USE_OPERATOR_MEMBERS();
  using Operator<CPUBackend>::RunImpl;

 private:
  std::vector<Displacement> displace_;
  DALIInterpType interp_type_;
  float fill_value_;

  bool has_mask_;
  const Tensor<CPUBackend> *mask_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DISPLACEMENT_DISPLACEMENT_FILTER_IMPL_CPU_H_
