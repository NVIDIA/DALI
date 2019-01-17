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
#include <memory>
#include "dali/aux/optical_flow/optical_flow_adapter.h"

namespace dali {
namespace optical_flow {
using kernels::TensorView;

static float kTestValue = 666;
static int kTestDataSize = 2;

class StubOpticalFlow : public OpticalFlowAdapter {
 public:
  explicit StubOpticalFlow(OpticalFlowParams params) :
          OpticalFlowAdapter(params) {
  }


  void CalcOpticalFlow(TensorView<GPUBackend, const uint8_t, 3> reference_image,
                       TensorView<GPUBackend, const uint8_t, 3> input_image,
                       TensorView<GPUBackend, float, 3> output_image,
                       TensorView<GPUBackend, const float, 3> external_hints) override {
    auto ptr = output_image.data;
    ptr[0] = kTestValue;
    ptr[1] = kTestValue / 2;
  }
};

TEST(OpticalFlowAdapter, StubApi) {
  std::unique_ptr<float> data(new float[kTestDataSize]);
  TensorView<GPUBackend, uint8_t, 3> tvref, tvin;
  TensorView<GPUBackend, float, 3> tvout(data.get(), {1, 1, 2});
  OpticalFlowParams params;
  std::unique_ptr<OpticalFlowAdapter> of(new StubOpticalFlow(params));
  of->CalcOpticalFlow(tvref, tvin, tvout);
  EXPECT_FLOAT_EQ(kTestValue, *tvout(0, 0, 0));
  EXPECT_FLOAT_EQ(kTestValue / 2, *tvout(0, 0, 1));
}

}  // namespace optical_flow
}  // namespace dali
