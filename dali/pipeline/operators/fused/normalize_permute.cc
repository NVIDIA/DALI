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

#include "dali/pipeline/operators/fused/normalize_permute.h"

namespace dali {

  template<>
  void NormalizePermute<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
    auto &input = ws->Input<CPUBackend>(idx);
    auto output = ws->Output<CPUBackend>(idx);

    DALI_ENFORCE(IsType<uint8>(input.type()));
    DALI_ENFORCE(input.ndim() == 3,
        "Expects 3-dim image input.");
    DALI_ENFORCE(input.dim(0) == H_,
        "Input image height does not match output height.");
    DALI_ENFORCE(input.dim(1) == W_,
          "Input image width does not match output width.");
    DALI_ENFORCE(input.dim(2) == C_,
        "Input image channels does not match output channels.");

    // Output is CHW
    output->Resize({C_, H_, W_});
    output->SetLayout(DALI_NCHW);
    if (output_type_ == DALI_FLOAT) {
      CPURunHelper<float>(input, output);
    } else {
      DALI_FAIL("Unsupported output type.");
    }
  }

template<>
template <typename OUT>
void NormalizePermute<CPUBackend>::CPURunHelper(const Tensor<CPUBackend> &input,
                                                Tensor<CPUBackend> *output) {
  const uint8 *in = input.template data<uint8>();
  OUT *out = output->template mutable_data<OUT>();
  float *mean = mean_.template mutable_data<float>();
  float *inv_std = inv_std_.template mutable_data<float>();

  for (int c = 0; c < C_; ++c) {
    for (int h = 0; h < H_; ++h) {
      for (int w = 0; w < W_; ++w) {
        out[c*H_*W_ + h*W_ + w] = static_cast<OUT>(
            (static_cast<float>(in[h*W_*C_ + w*C_ + c]) - mean[c]) * inv_std[c]);
      }
    }
  }
}

DALI_REGISTER_OPERATOR(NormalizePermute, NormalizePermute<CPUBackend>, CPU);

DALI_SCHEMA(NormalizePermute)
  .DocStr(R"code(Perform fused normalization, format conversion from NHWC to NCHW
and type casting.
Normalization takes input image and produces output using formula

..

output = (input - mean) / std
)code")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddOptionalArg("output_dtype",
      R"code(Output data type.)code", DALI_FLOAT)
  .AddOptionalArg("image_type",
        R"code(The color space of input and output image.)code", DALI_RGB)
  .AddArg("height",
      R"code(Height of the input image.)code",
      DALI_INT32)
  .AddArg("width",
      R"code(Width of the input image.)code",
      DALI_INT32)
  .AddArg("mean",
      R"code(Mean pixel values for image normalization.)code",
      DALI_FLOAT_VEC)
  .AddArg("std",
      R"code(Standard deviation values for image normalization.)code",
      DALI_FLOAT_VEC)
  .EnforceInputLayout(DALI_NHWC);

}  // namespace dali
