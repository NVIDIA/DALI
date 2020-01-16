// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/image/color/hsv.h"
#include "dali/kernels/imgproc/pointwise/linear_transformation_cpu.h"

namespace dali {
namespace {

template <typename Out, typename In>
using TheKernel = kernels::LinearTransformationCpu<Out, In, 3, 3, 3>;

}  // namespace

DALI_SCHEMA(Hsv)
    .DocStr(R"code(This operator performs HSV manipulation.
To change hue, saturation and/or value of the image, pass corresponding coefficients.
Keep in mind, that `hue` has additive delta argument,
while for `saturation` and `value` they are multiplicative.

This operator accepts RGB color space as an input.

For performance reasons, the operation is approximated by a linear transform in RGB space.
The color vector is projected along the neutral (gray) axis,
rotated (according to hue delta) and scaled according to value and saturation multiplers,
and then restored to original color space.
)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg(hsv::kHue,
                    R"code(Set additive change of hue. 0 denotes no-op)code",
                    0.0f, true)
    .AddOptionalArg(hsv::kSaturation,
                    R"code(Set multiplicative change of saturation. 1 denotes no-op)code",
                    1.0f, true)
    .AddOptionalArg(hsv::kValue,
                    R"code(Set multiplicative change of value. 1 denotes no-op)code",
                    1.0f, true)
    .AddOptionalArg(hsv::kOutputType, "Output data type; if not set, the input type is used.",
                    DALI_UINT8)
    .InputLayout(0, "HWC");


DALI_REGISTER_OPERATOR(Hsv, HsvCpu, CPU)


bool HsvCpu::SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  const auto &output = ws.template OutputRef<CPUBackend>(0);
  output_desc.resize(1);
  DetermineTransformation(ws);
  TYPE_SWITCH(input.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float, float16), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float, float16), (
          {
              using Kernel = TheKernel<OutputType, InputType>;
              kernel_manager_.Initialize<Kernel>();
              auto shapes = CallSetup<Kernel, InputType>(input);
              TypeInfo type;
              type.SetType<OutputType>(output_type_);
              output_desc[0] = {shapes, type};
          }
      ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())))  // NOLINT
  return true;
}


void HsvCpu::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  auto &output = ws.template OutputRef<CPUBackend>(0);
  output.SetLayout(InputLayout(ws, 0));
  auto &tp = ws.GetThreadPool();
  TYPE_SWITCH(input.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float, float16), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float, float16), (
          {
              using Kernel = TheKernel<OutputType, InputType>;
              for (int i = 0; i < input.shape().num_samples(); i++) {
                tp.DoWorkWithID([&, i](int thread_id) {
                  kernels::KernelContext ctx;
                  auto tvin = view<const InputType, 3>(input[i]);
                  auto tvout = view<OutputType, 3>(output[i]);
                  kernel_manager_.Run<Kernel>(ws.thread_idx(), i, ctx, tvout, tvin, tmatrices_[i]);
                });
              }
          }
      ), DALI_FAIL("Unsupported output type"))  // NOLINT
  ), DALI_FAIL("Unsupported input type"))  // NOLINT
  tp.WaitForWork();
}


}  // namespace dali
