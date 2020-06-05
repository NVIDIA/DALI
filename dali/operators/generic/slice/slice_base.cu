// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
  using Args = kernels::SliceArgs<OutputType, Dims>;

  explicit SliceBaseGpu(SliceBase<GPUBackend> &parent)
    : parent_(parent) {}

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<GPUBackend> &ws) override;
  void RunImpl(workspace_t<GPUBackend> &ws) override;

 private:
  SliceBase<GPUBackend> &parent_;
  std::vector<Args> args_;
  kernels::KernelManager kmgr_;
};

template <typename OutputType, typename InputType, int Dims>
bool SliceBaseGpu<OutputType, InputType, Dims>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                                          const workspace_t<GPUBackend> &ws) {
  parent_.FillArgs(args_, ws);
  const auto &input = ws.template InputRef<GPUBackend>(0);
  auto in_shape = input.shape();
  int nsamples = in_shape.num_samples();

  output_desc.resize(1);
  output_desc[0].type = TypeInfo::Create<OutputType>();
  output_desc[0].shape.resize(nsamples, Dims);

  kmgr_.Resize<Kernel>(1, 1);
  auto in_view = view<const InputType, Dims>(input);
  kernels::KernelContext ctx;
  auto req = kmgr_.Setup<Kernel>(0, ctx, in_view, args_);
  output_desc[0].shape = req.output_shapes[0];
  return true;
}

template <typename OutputType, typename InputType, int Dims>
void SliceBaseGpu<OutputType, InputType, Dims>::RunImpl(workspace_t<GPUBackend> &ws) {
  const auto &input = ws.template InputRef<GPUBackend>(0);
  auto &output = ws.template OutputRef<GPUBackend>(0);

  auto in_view = view<const InputType, Dims>(input);
  auto out_view = view<OutputType, Dims>(output);
  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  kmgr_.Run<Kernel>(0, 0, ctx, out_view, in_view, args_);
  output.SetLayout(input.GetLayout());
}

template <>
bool SliceBase<GPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const workspace_t<GPUBackend> &ws) {
  const auto &input = ws.template InputRef<GPUBackend>(0);
  auto input_type = input.type().id();
  auto ndim = input.shape().sample_dim();

  if (!impl_ || input_type_ != input_type || ndim != ndim_) {
    input_type_ = input_type;
    ndim_ = ndim;
    auto output_type = output_type_;
    if (output_type_ == DALI_NO_TYPE)
      output_type = input_type_;
    VALUE_SWITCH(ndim_, Dims, SLICE_DIMS, (
      TYPE_SWITCH(input_type_, type2id, InputType, SLICE_TYPES, (
        if (input_type_ == output_type) {
          using Impl = SliceBaseGpu<InputType, InputType, Dims>;
          impl_ = std::make_unique<Impl>(*this);
        } else {
          TYPE_SWITCH(output_type, type2id, OutputType, (float, float16, uint8_t), (
            using Impl = SliceBaseGpu<OutputType, InputType, Dims>;
            impl_ = std::make_unique<Impl>(*this);
          ), DALI_FAIL(make_string("Not supported output type:", output_type_));); // NOLINT
        }
      ), DALI_FAIL(make_string("Not supported input type: ", input_type_)););  // NOLINT
    ), DALI_FAIL(make_string("Not supported number of dimensions: ", ndim)););  // NOLINT
  }
  return impl_->SetupImpl(output_desc, ws);
}

template <>
void SliceBase<GPUBackend>::RunImpl(workspace_t<GPUBackend> &ws) {
  assert(impl_ != nullptr);
  impl_->RunImpl(ws);
}

}  // namespace dali
