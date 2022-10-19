// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <utility>
#include <vector>
#include <memory>
#include "dali/operators/signal/decibel/to_decibels_op.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel_params.h"
#include "dali/kernels/reduce/reduce_all_kernel_gpu.h"
#include "dali/kernels/signal/decibel/to_decibels_gpu.h"
#include "dali/pipeline/data/views.h"

namespace dali {

template <typename T>
class ToDecibelsImpl : public OpImplBase<GPUBackend> {
 public:
  using MaxKernel = kernels::reduce::ReduceAllGPU<T, T, kernels::reductions::max>;
  using ToDecibelsKernel = kernels::signal::ToDecibelsGpu<T>;
  using ToDecibelsArgs = kernels::signal::ToDecibelsArgs<T>;

  explicit ToDecibelsImpl(ToDecibelsArgs args)
      : args_(std::move(args)) {
    max_out_.SetContiguity(BatchContiguity::Contiguous);
    if (args_.ref_max) {
      kmgr_max_.Resize<MaxKernel>(1);
    }
    kmgr_todb_.Resize<ToDecibelsKernel>(1);
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;
  void RunImpl(Workspace &ws) override;

 private:
  ToDecibelsArgs args_;

  kernels::KernelManager kmgr_max_;
  kernels::KernelManager kmgr_todb_;

  std::vector<OutputDesc> max_out_desc_;
  TensorList<GPUBackend> max_out_;
};

template <typename T>
bool ToDecibelsImpl<T>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                  const Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto in_view = view<const T>(input);

  auto type = type2id<T>::value;

  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  if (args_.ref_max) {
    auto& req_max = kmgr_max_.Setup<MaxKernel>(0, ctx, in_view);
    max_out_desc_.resize(1);
    max_out_desc_[0].type = type;
    max_out_desc_[0].shape = req_max.output_shapes[0];
  }

  auto& req = kmgr_todb_.Setup<ToDecibelsKernel>(0, ctx, in_view);
  output_desc.resize(1);
  output_desc[0].type = type;
  output_desc[0].shape = req.output_shapes[0];

  return true;
}

template <typename T>
void ToDecibelsImpl<T>::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);

  auto in_view = view<const T>(input);
  auto out_view = view<T>(output);

  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();

  if (args_.ref_max) {
    max_out_.Resize(max_out_desc_[0].shape, max_out_desc_[0].type);
    auto max_values_view = view<T, 0>(max_out_);
    kmgr_max_.Run<MaxKernel>(0, ctx, max_values_view, in_view);
    kmgr_todb_.Run<ToDecibelsKernel>(0, ctx, out_view, in_view, args_, max_values_view);
  } else {
    kmgr_todb_.Run<ToDecibelsKernel>(0, ctx, out_view, in_view, args_);
  }
}

template <>
bool ToDecibels<GPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                       const Workspace &ws) {
  output_desc.resize(kNumOutputs);
  const auto &input = ws.Input<GPUBackend>(0);
  auto type = input.type();
  TYPE_SWITCH(type, type2id, T, (float), (
      using Impl = ToDecibelsImpl<T>;
      if (!impl_ || type != type_) {
        impl_ = std::make_unique<Impl>(args_);
        type_ = type;
      }
  ), DALI_FAIL(make_string("Unsupported data type: ", type)));  // NOLINT

  impl_->SetupImpl(output_desc, ws);
  return true;
}

template <>
void ToDecibels<GPUBackend>::RunImpl(Workspace &ws) {
  assert(impl_ != nullptr);
  impl_->RunImpl(ws);
}

DALI_REGISTER_OPERATOR(ToDecibels, ToDecibels<GPUBackend>, GPU);

}  // namespace dali
