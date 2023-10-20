// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace remap {
namespace test {

using namespace dali::remap::detail;  // NOLINT
using namespace std;  // NOLINT

template<typename T>
class RemapTest : public ::testing::Test {
 protected:
  void SetUp() final {
    uniform_real_distribution<> dist{0, 1000};
    auto rng = [&]() { return dist(mt_); };
    ref_data_.resize(data_shape_.num_elements());
    generate(ref_data_.begin(), ref_data_.end(), rng);

    test_data_.reshape(data_shape_);
    auto td_v = test_data_.cpu();
    std::copy(ref_data_.begin(), ref_data_.begin() + td_v.num_elements(), td_v[0].data);
  }

  vector<T> ref_data_;
  TensorListShape<> data_shape_{
          {240,  320},
          {480,  640},
          {1080, 1920},
          {10,   15},
  };
  cudaStream_t stream_ = 0;
  mt19937 mt_;

  dali::kernels::TestTensorList<T> test_data_;
};

using RemapTestTypes = ::testing::Types<float>;
TYPED_TEST_SUITE(RemapTest, RemapTestTypes);

TYPED_TEST(RemapTest, ShiftPixelOriginTest) {
  for (auto &val : this->ref_data_) {
    val += .5f;
  }
  dali::kernels::DynamicScratchpad ds;
  ShiftPixelOrigin(this->test_data_.gpu(this->stream_), .5f, ds,
                   this->stream_);
  CUDA_CALL(cudaStreamSynchronize(this->stream_));
  this->test_data_.invalidate_cpu();
  this->test_data_.cpu();
  CUDA_CALL(cudaStreamSynchronize(this->stream_));
  for (int i = 0; i < this->data_shape_.num_elements(); i++) {
    EXPECT_FLOAT_EQ(this->test_data_.cpu()[0].data[i], this->ref_data_[i]);
  }
}

}  // namespace test
}  // namespace remap
}  // namespace dali
