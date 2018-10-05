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
#include "dali/util/half.hpp"

namespace dali {

DALI_SCHEMA(NormalizeBase)
    .DocStr(R"code(Base Schema for normalization operators.)code")
    .AddArg("mean",
            R"code(Mean pixel values for image normalization.)code",
            DALI_FLOAT_VEC)
    .AddArg("std",
            R"code(Standard deviation values for image normalization.)code",
            DALI_FLOAT_VEC);


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
    .AddArg("height", R"code(Height of the input image.)code", DALI_INT32)
    .AddArg("width", R"code(Width of the input image.)code", DALI_INT32)
    .AddParent("CastPermute")
    .AddParent("NormalizeBase")
    .EnforceInputLayout(DALI_NHWC);

template<>
void NormalizePermute<CPUBackend>::DataDependentSetup(SampleWorkspace *ws, const int idx) {
  auto &input = ws->Input<CPUBackend>(idx);
  auto output = ws->Output<CPUBackend>(idx);

  DALI_ENFORCE(IsType<uint8>(input.type()));
  CheckShape(input.shape());

  // Output is CHW
  output->Resize({C_, H_, W_});
  output->SetLayout(DALI_NCHW);
}

template<>
template<typename Out>
void NormalizePermute<CPUBackend>::RunHelper(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  auto output = ws->Output<CPUBackend>(idx);
  const uint8 *in = input.template data<uint8>();
  Out *out = output->template mutable_data<Out>();
  const float *mean = mean_.template mutable_data<float>();
  const float *inv_std = inv_std_.template mutable_data<float>();

  for (int c = 0; c < C_; ++c) {
    for (int h = 0; h < H_; ++h) {
      for (int w = 0; w < W_; ++w) {
        out[c * H_ * W_ + h * W_ + w] = static_cast<Out>(
          (static_cast<float>(in[h * W_ * C_ + w * C_ + c]) - mean[c]) * inv_std[c]);
      }
    }
  }
}

template<>
void NormalizePermute<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  RUN_IMPL_CPU(ws, idx);
}

DALI_REGISTER_OPERATOR(NormalizePermute, NormalizePermute<CPUBackend>, CPU);

}  // namespace dali
