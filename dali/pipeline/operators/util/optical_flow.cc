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
#include "dali/aux/optical_flow/optical_flow_stub.h"
#include "optical_flow.h"

namespace dali {

const std::string kPresetArgName = "preset";   // NOLINT
const std::string kOutputFormatArgName = "output_format";   // NOLINT
const std::string kEnableHintsArgName = "enable_hints";   // NOLINT

DALI_SCHEMA(OpticalFlow)
                .DocStr(R"code(Calculates the Optical Flow for sequence of images given as a input.
 Mandatory input for the operator is a sequence of frames.
 As as optional input, operator accepts external hints for OF calculation.
The output format of this operator matches output format of OF driver API)code")
                .NumInput(1, 2)
                .NumOutput(1)
                .AddOptionalArg(kPresetArgName, R"code(Setting quality level of OF calculation.
 0.0f ... 1.0f, where 1.0f is best quality, lowest speed)code", .0f, false)
                .AddOptionalArg(kOutputFormatArgName,
                                R"code(Setting grid size for output vector.)code", -1, false)
                .AddOptionalArg(kEnableHintsArgName,
                                R"code(enabling/disabling temporal hints for sequences longer than 2 images)code",
                                false, false);

DALI_REGISTER_OPERATOR(OpticalFlow, OpticalFlow<CPUBackend>, CPU);


template<>
OpticalFlow<CPUBackend>::OpticalFlow(const OpSpec &spec) :
        Operator<CPUBackend>(spec),
        quality_factor_(spec.GetArgument<
                std::remove_const<decltype(this->quality_factor_)>::type>("preset")),
        grid_size_(spec.GetArgument<
                std::remove_const<decltype(this->grid_size_)>::type>("output_format")),
        enable_hints_(spec.GetArgument<
                std::remove_const<decltype(this->enable_hints_)>::type>("enable_hints")),
        of_params_({}),
        optical_flow_(std::unique_ptr<optical_flow::OpticalFlowAdapter<ComputeBackend>>(
                new optical_flow::OpticalFlowStub<ComputeBackend>(of_params_))) {
}


//template<>
//OpticalFlow<GPUBackend>::OpticalFlow(const OpSpec &spec) :
//        Operator<GPUBackend>(spec),
//        quality_factor_(spec.GetArgument<
//                std::remove_const<decltype(this->quality_factor_)>::type>("preset")),
//        grid_size_(spec.GetArgument<
//                std::remove_const<decltype(this->grid_size_)>::type>("output_format")),
//        enable_hints_(spec.GetArgument<
//                std::remove_const<decltype(this->enable_hints_)>::type>("enable_hints")),
//        of_params_({}),
//        optical_flow_(std::unique_ptr<optical_flow::OpticalFlowAdapter<kernels::ComputeGPU>>(
//                new optical_flow::OpticalFlowStub<kernels::ComputeGPU>(of_params_))) {
//}


template<>
void OpticalFlow<CPUBackend>::RunImpl(Workspace<CPUBackend> *ws, const int idx) {
  const auto &input = ws->template Input<CPUBackend>(idx);
//  auto tvin = view<const uint8_t, 3>(input);
//
  auto &output = ws->template Output<CPUBackend>(idx);
  output.ResizeLike(input);
  output.template mutable_data<uint8_t>();
//  auto tvout = view<float, 3>(output);
//
//  optical_flow_->CalcOpticalFlow(tvin[0], tvin[0], tvout[0]);
}

}  // namespace dali