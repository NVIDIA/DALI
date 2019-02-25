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
#include <fstream>
#include <cuda_runtime.h>
#include <dali/aux/optical_flow/turing_of/optical_flow_turing.h>
#include <dali/test/dali_test_config.h>
#include <dali/util/cuda_utils.h>


namespace dali {
namespace optical_flow {
namespace testing {

auto test_data_path = "/home/mszolucha/workspace/DALI_extra/db/";
//auto test_data_path = dali::testing::program_options::dali_extra_path();


TEST(OpticalFlowTuringTest, ExtractBitsTest) {
  ASSERT_EQ(9, kernel::extract_bits(3623, 2, 4));
  ASSERT_EQ(3328, kernel::extract_bits(-1535, 1, 12));
}


TEST(OpticalFlowTuringTest, CountDigitsTest) {
  ASSERT_EQ(1, kernel::count_digits(0));
  ASSERT_EQ(1, kernel::count_digits(1));
  ASSERT_EQ(2, kernel::count_digits(10));
  ASSERT_EQ(2, kernel::count_digits(15));
  ASSERT_EQ(2, kernel::count_digits(99));
  ASSERT_EQ(3, kernel::count_digits(100));
  ASSERT_EQ(3, kernel::count_digits(999));
  ASSERT_EQ(4, kernel::count_digits(1000));
  ASSERT_EQ(4, kernel::count_digits(9999));
  ASSERT_EQ(5, kernel::count_digits(10000));
  ASSERT_EQ(5, kernel::count_digits(99999));
  ASSERT_EQ(1, kernel::count_digits(-1));
  ASSERT_EQ(2, kernel::count_digits(-10));
  ASSERT_EQ(2, kernel::count_digits(-15));
  ASSERT_EQ(2, kernel::count_digits(-99));
  ASSERT_EQ(3, kernel::count_digits(-100));
  ASSERT_EQ(3, kernel::count_digits(-999));
  ASSERT_EQ(4, kernel::count_digits(-1000));
  ASSERT_EQ(4, kernel::count_digits(-9999));
  ASSERT_EQ(5, kernel::count_digits(-10000));
  ASSERT_EQ(5, kernel::count_digits(-99999));
}


TEST(OpticalFlowTuringTest, DecodeFlowVectorTest) {
  using namespace std;
  vector<int16_t> test_data = {101, -32376, 676, 3453, -23188};
  vector<float> ref_data = {3.5f, -12.8f, 21.4f, 107.29f, -299.12f};
  for (size_t i = 0; i < test_data.size(); i++) {
    EXPECT_EQ(ref_data[i], kernel::decode_flow_component(test_data[i]));
  }
}


TEST(OpticalFlowTuringTest, CudaDecodeFlowVectorTest) {
  using namespace std;
  ifstream infile(
          test_data_path + string("/optical_flow/flow_vector_decode/flow_vector_test_data.txt")),
          reffile(
          test_data_path + string("/optical_flow/flow_vector_decode/flow_vector_ground_truth.txt"));
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
  int16_t *incuda;
  float *outcuda;
  CUDA_CALL(cudaMallocManaged(&incuda, test_data.size() * sizeof(int16_t)));
  CUDA_CALL(cudaMallocManaged(&outcuda, ref_data.size() * sizeof(float)));
  CUDA_CALL(cudaMemcpy(incuda, test_data.data(), test_data.size() * sizeof(int16_t),
                       cudaMemcpyDefault));
  kernel::DecodeFlowComponents(incuda, outcuda, test_data.size());
  for (size_t i = 0; i < ref_data.size(); i++) {
    EXPECT_EQ(ref_data[i], outcuda[i]);
  }
  CUDA_CALL(cudaFree(incuda));
  CUDA_CALL(cudaFree(outcuda));
}


}  // namespace testing
}  // namespace optical_flow

}  // namespace dali

