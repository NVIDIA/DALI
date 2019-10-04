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

#include "dali/pipeline/operators/color/hsv.h"
#include "dali/kernels/imgproc/pointwise/linear_transformation_cpu.h"

namespace dali {
namespace {


template <typename Out, typename In>
using TheKernel = kernels::LinearTransformationCpu<Out, In, 3, 3, 3>;

}  // namespace


DALI_SCHEMA(Hsv)
                .DocStr(R"code()code")
                .NumInput(1)
                .NumOutput(1)
                .AddOptionalArg(hsv::kHue,
                                R"code(Set additive brightness delta. 0 denotes no-op)code", .0f,
                                true)
                .AddOptionalArg(hsv::kSaturation,
                                R"code(Set multiplicative contrast delta. 1 denotes no-op)code",
                                1.f, true)
                .AddOptionalArg(hsv::kValue,
                                R"code(Set multiplicative contrast delta. 1 denotes no-op)code",
                                1.f, true)
                .AddOptionalArg(hsv::kOutputType, R"code(Set output data type)code", DALI_UINT8);

DALI_REGISTER_OPERATOR(Hsv, HsvCpu, CPU)


bool HsvCpu::SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                       const workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  const auto &output = ws.template OutputRef<CPUBackend>(0);
  output_desc.resize(1);
  // @autoformat:off
  TYPE_SWITCH(input.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float, float16), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float, float16), (
          {
              using Kernel = TheKernel<OutputType, InputType>;
              kernel_manager_.Initialize<Kernel>();
              auto shapes = CallSetup<Kernel, InputType>(input, ws.data_idx());
              TypeInfo type;
              type.SetType<OutputType>(output_type_);
              output_desc[0] = {shapes, type};
          }
      ), DALI_FAIL(make_string("Unsupported output type:", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type:", input.type().id())))  // NOLINT
  // @autoformat:on
  return true;
}


void HsvCpu::RunImpl(Workspace<CPUBackend> &ws) {
  const auto &input = ws.template Input<CPUBackend>(0);
  auto &output = ws.template Output<CPUBackend>(0);
  // @autoformat:off
  TYPE_SWITCH(input.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float, float16), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float, float16), (
          {
              using Kernel = TheKernel<OutputType, InputType>;
              kernels::KernelContext ctx;
              auto tvin = view<const InputType, 3>(input);
              auto tvout = view<OutputType, 3>(output);
              auto tmat = transformation_matrix(hue_, saturation_, value_);
              kernel_manager_.Run<Kernel>(ws.thread_idx(), ws.data_idx(), ctx, tvout, tvin, tmat);
          }
      ), DALI_FAIL("Unsupported output type"))  // NOLINT
  ), DALI_FAIL("Unsupported input type"))  // NOLINT
  // @autoformat:on
}


}  // namespace dali
