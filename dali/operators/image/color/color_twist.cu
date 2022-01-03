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

#include <vector>
#include "dali/operators/image/color/color_twist.h"
#include "dali/kernels/imgproc/pointwise/linear_transformation_gpu.h"

namespace dali {
namespace {

template <typename Out, typename In>
using TheKernel = kernels::LinearTransformationGpu<Out, In, 3, 3, 2>;

}  // namespace

DALI_REGISTER_OPERATOR(Hsv, ColorTwistGpu, GPU)
DALI_REGISTER_OPERATOR(Hue, ColorTwistGpu, GPU);
DALI_REGISTER_OPERATOR(Saturation, ColorTwistGpu, GPU);
DALI_REGISTER_OPERATOR(ColorTwist, ColorTwistGpu, GPU);

bool ColorTwistGpu::SetupImpl(std::vector<OutputDesc> &output_desc, const DeviceWorkspace &ws) {
  KMgrResize(num_threads_, max_batch_size_);
  const auto &input = ws.template Input<GPUBackend>(0);
  output_desc.resize(1);
  DetermineTransformation(ws);
  auto sh = input.shape();
  auto layout = input.GetLayout();
  assert(ImageLayoutInfo::IsChannelLast(layout));
  TYPE_SWITCH(input.type(), type2id, InputType, COLOR_TWIST_SUPPORTED_TYPES, (
    TYPE_SWITCH(output_type_, type2id, OutputType, COLOR_TWIST_SUPPORTED_TYPES, (
      {
        using Kernel = TheKernel<OutputType, InputType>;
        kernel_manager_.Initialize<Kernel>();
        kernels::KernelContext ctx;
        ctx.gpu.stream = ws.stream();
        auto ndim = sh.sample_dim();
        DALI_ENFORCE(ndim >= 3 && ndim <= 4, "Unsupported number of dims");
        const auto tvin = ndim == 3 ? view<const InputType, 3>(input) :
                          reinterpret<const InputType, 3>(view<const InputType, 4>(input),
                                            collapse_dim(view<const InputType, 4>(input).shape, 0),
                                                         true);
        const auto &reqs = kernel_manager_.Setup<Kernel>(0, ctx, tvin, make_cspan(tmatrices_),
                                                        make_cspan(toffsets_));
        output_desc[0] = {input.shape(), output_type_};
      }
    ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type())))  // NOLINT
  return true;
}

template <typename OutputType, typename InputType>
void ColorTwistGpu::RunImplHelper(workspace_t<GPUBackend> &ws) {
  const auto &input = ws.template Input<GPUBackend>(0);
  auto &output = ws.template Output<GPUBackend>(0);
  output.SetLayout(input.GetLayout());
  auto sh = input.shape();
  auto num_dims = sh.sample_dim();
  using Kernel = TheKernel<OutputType, InputType>;
  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  auto tvin = num_dims == 3 ? view<const InputType, 3>(input) :
                              reinterpret<const InputType, 3>(view<const InputType, 4>(input),
                                collapse_dim(view<const InputType, 4>(input).shape, 0), true);
  auto tvout = num_dims == 3 ? view<OutputType, 3>(output):
                                reinterpret<OutputType, 3>(view<OutputType, 4>(output),
                                  collapse_dim(view<OutputType, 4>(output).shape, 0), true);
  kernel_manager_.Run<Kernel>(0, 0, ctx, tvout, tvin,
                                make_cspan(tmatrices_), make_cspan(toffsets_));
}

void ColorTwistGpu::RunImpl(workspace_t<GPUBackend> &ws) {
  const auto &input = ws.template Input<GPUBackend>(0);
  TYPE_SWITCH(input.type(), type2id, InputType, COLOR_TWIST_SUPPORTED_TYPES, (
    TYPE_SWITCH(output_type_, type2id, OutputType, COLOR_TWIST_SUPPORTED_TYPES, (
      {
        RunImplHelper<OutputType, InputType>(ws);
      }
    ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type())))  // NOLINT
}


}  // namespace dali
