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

#include "optical_flow_adapter.h"
#include <dali/kernels/tensor_view.h>

#ifndef DALI_OPTICAL_FLOW_STUB_H
#define DALI_OPTICAL_FLOW_STUB_H

namespace dali {
namespace optical_flow {
using kernels::TensorView;

class OpticalFlowStub : public OpticalFlowAdapter {
public:
  explicit OpticalFlowStub(OpticalFlowParams params) :OpticalFlowAdapter(params){}


  void CalcOpticalFlow(TensorView<GPUBackend, const uint8_t, 3> reference_image,
                       TensorView<GPUBackend, const uint8_t, 3> input_image,
                       TensorView<GPUBackend, float, 3> output_image,
                       TensorView<GPUBackend, const float, 3> external_hints) override {}

  static constexpr float kStubValue = 666.f;
};

}  // namespace optical_flow
}  // namespace dali

#endif  // DALI_OPTICAL_FLOW_STUB_H
