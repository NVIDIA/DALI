// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  const auto &input = ws.template InputRef<GPUBackend>(0);
  output_desc.resize(1);
  DetermineTransformation(ws);
  TYPE_SWITCH(input.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
          {
              using Kernel = TheKernel<OutputType, InputType>;
              kernel_manager_.Initialize<Kernel>();
              auto &shapes = CallSetup<Kernel, InputType>(ws, input);
              TypeInfo type;
              type.SetType<OutputType>(output_type_);
              output_desc[0] = {shapes, type};
          }
      ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())))  // NOLINT
  return true;
}


void ColorTwistGpu::RunImpl(workspace_t<GPUBackend> &ws) {
  const auto &input = ws.template Input<GPUBackend>(0);
  auto &output = ws.template Output<GPUBackend>(0);
  output.SetLayout(input.GetLayout());
  TYPE_SWITCH(input.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
          {
              using Kernel = TheKernel<OutputType, InputType>;
              kernels::KernelContext ctx;
              ctx.gpu.stream = ws.stream();
              auto tvin = view<const InputType, 3>(input);
              auto tvout = view<OutputType, 3>(output);
              kernel_manager_.Run<Kernel>(ws.thread_idx(), 0, ctx, tvout, tvin,
                                          make_cspan(tmatrices_), make_cspan(toffsets_));
          }
      ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())))  // NOLINT
}


}  // namespace dali
