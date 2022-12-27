// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/backend_tags.h"
#include "dali/operators/sequence/optical_flow/optical_flow_adapter/optical_flow_stub.h"
#include "dali/core/cuda_error.h"

namespace dali {
namespace optical_flow {

static int kTestDataSize = 2;

TEST(OpticalFlowAdapter, StubApiCpuBackend) {
  TensorView<StorageCPU, uint8_t, 3> tvref, tvin;
  std::vector<float> in_data(kTestDataSize);
  TensorView<StorageCPU, float, 3> tvout(in_data.data(), {1, 1, 2});
  OpticalFlowParams params = {0.f, 4, 4, false, false};
  std::unique_ptr<OpticalFlowAdapter<ComputeCPU>> of(new OpticalFlowStub<ComputeCPU>(params));
  of->CalcOpticalFlow(tvref, tvin, tvout);
  EXPECT_FLOAT_EQ(OpticalFlowStub<ComputeCPU>::kStubValue, *tvout(0, 0, 0));
  EXPECT_FLOAT_EQ(OpticalFlowStub<ComputeCPU>::kStubValue / 2, *tvout(0, 0, 1));
  auto ts = of->CalcOutputShape(1, 1);
  TensorShape<3> ref_ts{2, 3, 4};
  of->Prepare(1, 1);
  EXPECT_EQ(ref_ts, ts);
}


TEST(OpticalFlowAdapter, StubApiGpuBackend) {
  using StubValueType = std::remove_const_t<decltype(OpticalFlowStub<ComputeGPU>::kStubValue)>;
  StubValueType *tvout_data;
  CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&tvout_data),
                       kTestDataSize * sizeof(StubValueType)));

  TensorView<StorageGPU, uint8_t, 3> tvref, tvin;
  TensorView<StorageGPU, float, 3> tvout(tvout_data, {1, 1, 2});
  OpticalFlowParams params = {0.f, 4, 4, false, false};
  std::unique_ptr<OpticalFlowAdapter<ComputeGPU>> of(new OpticalFlowStub<ComputeGPU>(params));
  of->CalcOpticalFlow(tvref, tvin, tvout);

  std::vector<float> host(kTestDataSize);
  CUDA_CALL(cudaMemcpy(host.data(), tvout.data, kTestDataSize * sizeof(StubValueType),
                       cudaMemcpyDeviceToHost));
  EXPECT_FLOAT_EQ(OpticalFlowStub<ComputeGPU>::kStubValue, host[0]);
  EXPECT_FLOAT_EQ(OpticalFlowStub<ComputeGPU>::kStubValue / 2, host[1]);

  CUDA_CALL(cudaFree(tvout_data));

  auto ts = of->CalcOutputShape(1, 1);
  TensorShape<3> ref_ts{2, 3, 4};
  of->Prepare(1, 1);
  EXPECT_EQ(ref_ts, ts);
}

}  // namespace optical_flow
}  // namespace dali
