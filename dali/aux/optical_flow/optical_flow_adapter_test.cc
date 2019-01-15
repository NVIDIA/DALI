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

#include <gtest/gtest.h>
#include "dali/aux/optical_flow/optical_flow_adapter.h"

namespace dali {
namespace optical_flow {
using namespace kernels;

class StubOpticalFlow : public OpticalFlowAdapter {
 public:
  void CalcOpticalFlow(const TensorView<GPUBackend, uint8_t, 3> reference_image,
                       const TensorView<GPUBackend, uint8_t, 3> input_image,
                       TensorView<GPUBackend, float, 1> output_image,
                       const TensorView<GPUBackend, float, 1> external_hints) override {

  }
};

TEST(OpticalFlowAdapter, StubApi) {
  TensorView<GPUBackend, uint8_t, 3> tvref, tvin;
  TensorView<GPUBackend, float, 1> tvout;
  std::unique_ptr<OpticalFlowAdapter> of(new StubOpticalFlow());
  OpticalFlowParams params;
  of->Initialize(params);
  of->CalcOpticalFlow(tvref, tvin, tvout);
}

}  // namespace optical_flow
}  // namespace dali