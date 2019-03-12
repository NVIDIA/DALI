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
#include <opencv2/opencv.hpp>
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
  // (In)sanity test
  using std::vector;
  vector<int16_t> test_data = {101, -32376, 676, 3453, -23188};
  vector<float> ref_data = {3.15625f, -1011.75f, 21.125f, 107.90625f, -724.625f};
  for (size_t i = 0; i < test_data.size(); i++) {
    EXPECT_NEAR(ref_data[i], kernel::decode_flow_component(test_data[i]), kFlowVectorEpsilon);
  }
}


TEST(OpticalFlowTuringTest, BgrToAbgrSynteticTest) {
  std::vector<uint8_t> data = {
          73, 5, 47, 71, 30, 1,
          80, 41, 60, 60, 85, 41,
          55, 66, 4, 94, 59, 47,
          64, 83, 96, 61, 30, 95,
          88, 95, 63, 96, 78, 16,
          81, 89, 81, 2, 18, 35,
  };
  std::vector<uint8_t> reference = {
          255, 73, 5, 47, 255, 71, 30, 1, 0, 0,
          255, 80, 41, 60, 255, 60, 85, 41, 0, 0,
          255, 55, 66, 4, 255, 94, 59, 47, 0, 0,
          255, 64, 83, 96, 255, 61, 30, 95, 0, 0,
          255, 88, 95, 63, 255, 96, 78, 16, 0, 0,
          255, 81, 89, 81, 255, 2, 18, 35, 0, 0,
  };
  size_t width = 2, pitch = 10, height = 6;

  uint8_t *input, *tested;
  CUDA_CALL(cudaMallocManaged(&input, data.size()));
  CUDA_CALL(cudaMallocManaged(&tested, reference.size()));
  CUDA_CALL(cudaMemcpy(input, data.data(), data.size(), cudaMemcpyDefault));

  kernel::BgrToAbgr(input, tested, pitch, width, height);
  cudaDeviceSynchronize();

  for (size_t i = 0; i < reference.size(); i++) {
    EXPECT_EQ(reference[i], tested[i]) << "Failed on index: " << i;
  }

  CUDA_CALL(cudaFree(tested));
  CUDA_CALL(cudaFree(input));
}

// DISABLED due to lack of test data. Enable on next possible chance
TEST(OpticalFlowTuringTest, DISABLED_CudaDecodeFlowVectorTest) {
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
  kernel::DecodeFlowComponents(static_cast<int16_t *> (incuda), outcuda,
                               inpitch, inwidth, inheight);
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

