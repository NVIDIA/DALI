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

#include <dali/pipeline/data/views.h>
#include <dali/pipeline/operators/util/optical_flow.h>

namespace dali {


DALI_SCHEMA(OpticalFlow)
                .DocStr(R"code(Calculates the Optical Flow for sequence of images given as a input.
Mandatory input for the operator is a sequence of frames.
As an optional input, operator accepts external hints for OF calculation.
The output format of this operator matches the output format of OF driver API.
Dali uses Turing optical flow hardware implementation: https://developer.nvidia.com/opticalflow-sdk
)code")
                .NumInput(1, 2)
                .NumOutput(1)
                .AddOptionalArg(detail::kPresetArgName, R"code(Setting quality level of OF calculation.
 0.0f ... 1.0f, where 1.0f is best quality, lowest speed)code", .0f, false)
                .AddOptionalArg(detail::kOutputFormatArgName,
                                R"code(Setting grid size for output vector.
Value defines width of grid square (e.g. if value == 4, 4x4 grid is used).
For values <=0, grid size is undefined. Currently only grid_size=4 is supported.)code", -1, false)
                .AddOptionalArg(detail::kEnableHintsArgName,
                                R"code(enabling/disabling temporal hints for sequences longer than 2 images)code",
                                false, false);


DALI_REGISTER_OPERATOR(OpticalFlow, OpticalFlow<GPUBackend>, GPU);


template<>
void OpticalFlow<GPUBackend>::RunImpl(Workspace<GPUBackend> *ws, const int) {
  if (enable_hints_) {
    const auto &input = ws->Input<GPUBackend>(0);
    const auto &external_hints = ws->Input<GPUBackend>(1);
    auto &output = ws->Output<GPUBackend>(0);

    output.ResizeLike(input);

    auto in = view<const uint8_t, 3>(input);
    auto hints = view<const float, 3>(external_hints);
    auto out = view<float, 3>(output);
    DALI_ENFORCE(in.size() == out.size(), "Number of tensors in TensorList don't match.");

    for (decltype(in.size()) i = 1; i < in.size(); i++) {
      optical_flow_->CalcOpticalFlow(in[i - 1], in[i], out[i - 1], hints[i]);
    }
  } else {
    const auto &input = ws->Input<GPUBackend>(0);
    auto &output = ws->Output<GPUBackend>(0);

    output.ResizeLike(input);

    auto in = view<const uint8_t, 3>(input);
    auto out = view<float, 3>(output);
    DALI_ENFORCE(in.size() == out.size(), "Number of tensors in TensorList don't match.");

    for (decltype(in.size()) i = 1; i < in.size(); i++) {
      optical_flow_->CalcOpticalFlow(in[i - 1], in[i], out[i - 1]);
    }
  }
}

}  // namespace dali
