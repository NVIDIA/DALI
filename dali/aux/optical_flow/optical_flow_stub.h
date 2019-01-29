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

#ifndef DALI_AUX_OPTICAL_FLOW_OPTICAL_FLOW_STUB_H_
#define DALI_AUX_OPTICAL_FLOW_OPTICAL_FLOW_STUB_H_

#include "dali/kernels/tensor_view.h"
#include "dali/aux/optical_flow/optical_flow_adapter.h"

namespace dali {
namespace optical_flow {

/**
 * Stub implementation for OpticalFlow.
 * All it does is assign two values: 666.f and 333.f to output_image
 */
class DLL_PUBLIC OpticalFlowStub : public OpticalFlowAdapter {
 public:
  explicit OpticalFlowStub(OpticalFlowParams params);

  void CalcOpticalFlow(TV<detail::Backend, const uint8_t, 3> reference_image,
                       TV<detail::Backend, const uint8_t, 3> input_image,
                       TV<detail::Backend, float, 3> output_image,
                       TV<detail::Backend, const float, 3> external_hints) override;

  static constexpr float kStubValue = 666.f;
};

}  // namespace optical_flow
}  // namespace dali

#endif  // DALI_AUX_OPTICAL_FLOW_OPTICAL_FLOW_STUB_H_
