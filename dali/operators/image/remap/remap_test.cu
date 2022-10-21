// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/image/remap/remap.h"
#include "dali/operators/image/remap/remap.cuh"
#include <gtest/gtest.h>
#include "dali/test/tensor_test_utils.h"

namespace dali::remap::test {
using namespace dali::remap::detail;

using namespace std;  // NOLINT

template<typename T>
class ShiftPixelOriginTest : public ::testing::Test {
 protected:
  void SetUp() final {
    uniform_real_distribution<> dist{0, 1000};
    auto rng = [&]() { return dist(mt_); };
    ref_data_.resize(data_shape_.num_elements());
    generate(ref_data_.begin(), ref_data_.end(), rng);
    cudaMallocManaged(&test_data_, data_shape_.num_elements() * sizeof(T));
    CUDA_CALL(cudaMemcpy(test_data_, ref_data_.data(), data_shape_.num_elements() * sizeof(T),
                         cudaMemcpyDefault));
  }


  void TearDown() final {
    cudaFree(test_data_);
  }


  T *test_data_ = nullptr;
  vector<T> ref_data_;
  TensorListShape<> data_shape_{{240,  320},
                                {1080, 1920},
                                {480,  640}};
  cudaStream_t stream_ = 0;
  mt19937 mt_;
};

using ShiftPixelOriginTestTypes = ::testing::Types<float>;
TYPED_TEST_SUITE(ShiftPixelOriginTest, ShiftPixelOriginTestTypes);

TYPED_TEST(ShiftPixelOriginTest, ShiftPixelOriginTest) {
  using T = TypeParam;
  for (auto &val: this->ref_data_) {
    val += .5f;
  }
  ShiftPixelOrigin(TensorListView<StorageUnified, T>(this->test_data_, this->data_shape_),
                   this->stream_);
  CUDA_CALL(cudaStreamSynchronize(this->stream_));
  for (int i = 0; i < this->data_shape_.num_elements(); i++) {
    EXPECT_FLOAT_EQ(this->test_data_[i], this->ref_data_[i]);
  }
}

}  // namespace dali
