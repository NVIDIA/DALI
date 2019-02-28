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

#include <dali/aux/optical_flow/turing_of/optical_flow_turing.h>
#include <dali/test/dali_test_config.h>
#include <dali/util/cuda_utils.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <string>

namespace dali {
namespace optical_flow {
namespace testing {

namespace {
constexpr float kFlowVectorEpsilon = 1.f / 32;
}

TEST(OpticalFlowTuringTest, DecodeFlowVectorTest) {
  using std::vector;
  vector<int16_t> test_data = {101, -32376, 676, 3453, -23188};
  vector<float> ref_data = {3.15625f, -12.25f, 21.125f, 107.90625f, -299.375f};
  for (size_t i = 0; i < test_data.size(); i++) {
    EXPECT_NEAR(ref_data[i], kernel::decode_flow_component(test_data[i]), kFlowVectorEpsilon);
  }
}


TEST(OpticalFlowTuringTest, CudaDecodeFlowVectorTest) {
  using std::ifstream;
  using std::string;
  using std::vector;
  auto test_data_path = dali::testing::dali_extra_path() + "/db/optical_flow/flow_vector_decode/";
  ifstream infile(test_data_path + "flow_vector_test_data.txt"),
          reffile(test_data_path + "flow_vector_ground_truth.txt");
  ASSERT_TRUE(infile && reffile) << "Error accessing test data";
  vector<int16_t> test_data;
  int16_t valin;
  float valref;
  while (infile >> valin) {
    test_data.push_back(valin);
  }
  vector<float> ref_data;
  while (reffile >> valref) {
    ref_data.push_back(valref);
  }
  ASSERT_EQ(ref_data.size(), test_data.size());
  void *incuda;
  float *outcuda;
  size_t inpitch;

  size_t inwidth = 200, inheight = 50;
  ASSERT_EQ(test_data.size(), inwidth * inheight) << "Provided dims don't match with data size";

  CUDA_CALL(cudaMallocPitch(&incuda, &inpitch, inwidth * sizeof(int16_t), inheight));
  CUDA_CALL(cudaMemcpy2D(incuda, inpitch, test_data.data(), inwidth * sizeof(int16_t),
                         inwidth * sizeof(int16_t), inheight, cudaMemcpyDefault));
  CUDA_CALL(cudaMallocManaged(&outcuda, ref_data.size() * sizeof(float)));
  kernel::DecodeFlowComponents(reinterpret_cast<int16_t *> (incuda), outcuda,
                               inpitch / sizeof(int16_t), inwidth, inheight);
  CUDA_CALL(cudaDeviceSynchronize());
  for (size_t i = 0; i < ref_data.size(); i++) {
    EXPECT_NEAR(ref_data[i], outcuda[i], kFlowVectorEpsilon) << "Failed on idx: " << i;
  }
  CUDA_CALL(cudaFree(incuda));
  CUDA_CALL(cudaFree(outcuda));
}


}  // namespace testing
}  // namespace optical_flow

}  // namespace dali

