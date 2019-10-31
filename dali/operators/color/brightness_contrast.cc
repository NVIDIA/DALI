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

#include "dali/operators/color/brightness_contrast.h"
#include "dali/kernels/imgproc/color_manipulation/brightness_contrast.h"

namespace dali {
namespace {

template <typename Out, typename In>
using TheKernel = kernels::BrightnessContrastCpu<Out, In, 3>;

}  // namespace


DALI_SCHEMA(BrightnessContrast)
                .DocStr(R"code(Change the brightness and contrast of the image.
Additionally, this operator can change the type of data.)code")
                .NumInput(1)
                .NumOutput(1)
                .AddOptionalArg(brightness_contrast::kBrightness,
                                R"code(Set additive brightness delta. 0 denotes no-op)code", .0f,
                                true)
                .AddOptionalArg(brightness_contrast::kContrast,
                                R"code(Set multiplicative contrast delta. 1 denotes no-op)code",
                                1.f, true)
                .AddOptionalArg(brightness_contrast::kOutputType,
                                R"code(Set output data type)code", DALI_INT16);

DALI_REGISTER_OPERATOR(BrightnessContrast, BrightnessContrastCpu, CPU)


bool BrightnessContrastCpu::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  const auto &output = ws.template OutputRef<CPUBackend>(0);
  output_desc.resize(1);
  AcquireArguments(ws);
  TYPE_SWITCH(input.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
          {
              using Kernel = TheKernel<OutputType, InputType>;
              kernel_manager_.Initialize<Kernel>();
              auto shapes = CallSetup<Kernel, InputType>(input, ws.data_idx());
              TypeInfo type;
              type.SetType<OutputType>(output_type_);
              output_desc[0] = {shapes, type};
          }
      ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())))  // NOLINT
  return true;
}


void BrightnessContrastCpu::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  auto &output = ws.template OutputRef<CPUBackend>(0);
  TYPE_SWITCH(input.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
          {
              using Kernel = TheKernel<OutputType, InputType>;
              kernels::KernelContext ctx;
              auto& tp = ws.GetThreadPool();
              for (int sample_id = 0; sample_id < input.shape().num_samples(); sample_id++) {
                tp.DoWorkWithID([&, sample_id](int thread_id) {
                    auto tvin = view<const InputType, 3>(input[sample_id]);
                    auto tvout = view<OutputType, 3>(output[sample_id]);
                    kernel_manager_.Run<Kernel>(thread_id, sample_id, ctx, tvout, tvin,
                                                brightness_[sample_id], contrast_[sample_id]);
                });
              }
          }
      ), DALI_FAIL("Unsupported output type"))  // NOLINT
  ), DALI_FAIL("Unsupported input type"))  // NOLINT
}


}  // namespace dali
