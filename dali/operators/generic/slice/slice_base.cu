// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include <memory>
#include "dali/operators/generic/slice/slice_base.h"
#include "dali/kernels/slice/slice_gpu.cuh"

namespace dali {

template <typename OutputType, typename InputType, int Dims>
class SliceBaseGpu : public OpImplBase<GPUBackend> {
 public:
  using Kernel = kernels::SliceGPU<OutputType, InputType, Dims>;
  using SliceArgs = kernels::SliceArgs<OutputType, Dims>;

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;
  void RunImpl(Workspace &ws) override;

  std::vector<SliceArgs>& Args() { return args_; }

 private:
  std::vector<SliceArgs> args_;
  kernels::KernelManager kmgr_;
};

template <typename OutputType, typename InputType, int Dims>
bool SliceBaseGpu<OutputType, InputType, Dims>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                                          const Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto in_shape = input.shape();
  int nsamples = in_shape.num_samples();

  output_desc.resize(1);
  output_desc[0].type = type2id<OutputType>::value;
  output_desc[0].shape.resize(nsamples, Dims);

  kmgr_.Resize<Kernel>(1);
  auto in_view = view<const InputType, Dims>(input);
  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  auto req = kmgr_.Setup<Kernel>(0, ctx, in_view, args_);
  output_desc[0].shape = req.output_shapes[0];
  return true;
}

template <typename OutputType, typename InputType, int Dims>
void SliceBaseGpu<OutputType, InputType, Dims>::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);

  auto in_view = view<const InputType, Dims>(input);
  auto out_view = view<OutputType, Dims>(output);
  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  kmgr_.Run<Kernel>(0, ctx, out_view, in_view, args_);
  output.SetLayout(input.GetLayout());
}

template <>
bool SliceBase<GPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto input_type = input.type();
  auto ndim = input.shape().sample_dim();

  if (!impl_ || input_type_ != input_type || ndim != ndim_) {
    impl_.reset();
    input_type_ = input_type;
    ndim_ = ndim;
  }
  auto output_type = output_type_ == DALI_NO_TYPE ? input_type_ : output_type_;

  VALUE_SWITCH(ndim_, Dims, SLICE_DIMS, (
    TYPE_SWITCH(input_type_, type2id, InputType, SLICE_TYPES, (
      if (input_type_ == output_type) {
        using Impl = SliceBaseGpu<InputType, InputType, Dims>;
        if (!impl_)
          impl_ = std::make_unique<Impl>();
        FillArgs(reinterpret_cast<Impl*>(impl_.get())->Args(), ws);
      } else {
        TYPE_SWITCH(output_type, type2id, OutputType, (float, float16, uint8_t), (
          using Impl = SliceBaseGpu<OutputType, InputType, Dims>;
          if (!impl_)
            impl_ = std::make_unique<Impl>();
          FillArgs(reinterpret_cast<Impl*>(impl_.get())->Args(), ws);
        ), DALI_FAIL(make_string("Not supported output type: ", output_type));); // NOLINT
      }
    ), DALI_FAIL(make_string("Not supported input type: ", input_type_)););  // NOLINT
  ), DALI_FAIL(make_string("Not supported number of dimensions: ", ndim)););  // NOLINT

  return impl_->SetupImpl(output_desc, ws);
}

template <>
void SliceBase<GPUBackend>::RunImpl(Workspace &ws) {
  assert(impl_ != nullptr);
  impl_->RunImpl(ws);
}

}  // namespace dali
