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
#include "dali/aux/optical_flow/optical_flow_stub.h"

namespace dali {
namespace optical_flow {

using kernels::TensorView;

static int kTestDataSize = 2;

TEST(OpticalFlowAdapter, StubApi) {
  std::unique_ptr<float> data(new float[kTestDataSize]);
  TensorView<detail::Backend, uint8_t, 3> tvref, tvin;
  TensorView<detail::Backend, float, 3> tvout(data.get(), {1, 1, 2});
  OpticalFlowParams params;
  std::unique_ptr<OpticalFlowAdapter> of(new OpticalFlowStub(params));
  of->CalcOpticalFlow(tvref, tvin, tvout);
  EXPECT_FLOAT_EQ(OpticalFlowStub::kStubValue, *tvout(0, 0, 0));
  EXPECT_FLOAT_EQ(OpticalFlowStub::kStubValue / 2, *tvout(0, 0, 1));
}

}  // namespace optical_flow
}  // namespace dali
