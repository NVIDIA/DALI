// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include <utility>
#include <random>
#include "dali/operators/audio/nonsilence_op.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/kernels/signal/decibel/decibel_calculator.h"

namespace dali {
namespace testing {

class NonsilenceOpTest : public ::testing::Test {
 protected:
  void SetUp() final {
    impl_.nsamples_ = 1;
  }


  std::vector<float> input_{0, 0, 0, 0, 1000, -1000, 1000, 0, 0, 0};
  int window_size_ = 3;
  std::vector<float> mms_ref_{0, 0, 333333.344, 666666.688, 1000000, 666666.688, 333333.344, 0};
  std::pair<int, int> nonsilence_region_{2, 5};
  int buffer_length_ = 10;
  TensorShape<1> shape_ = {buffer_length_};
  NonsilenceOperatorCpu::Impl impl_;
};


TEST_F(NonsilenceOpTest, UnderlyingKernelsTest) {
  auto &ns_op = this->impl_;
  auto in = make_tensor_cpu(reinterpret_cast<const float *>(this->input_.data()), this->shape_);
  kernels::signal::MovingMeanSquareArgs mms_args{this->window_size_, -1};
  ns_op.SetupKernel();
  ns_op.RunKernel(0, in, mms_args);

  auto &mms_output = ns_op.intermediate_buffers_;

  ASSERT_EQ(this->mms_ref_.size(), mms_output[0].size());
  for (size_t i = 0; i < this->mms_ref_.size(); i++) {
    EXPECT_FLOAT_EQ(this->mms_ref_[i], mms_output[0].data<float>()[i]);
  }
}


TEST_F(NonsilenceOpTest, DetectNonsilenceRegionTest) {
  auto &ns_op = this->impl_;
  auto in = make_tensor_cpu(reinterpret_cast<const float *>(this->input_.data()), this->shape_);
  auto nonsilence_region = ns_op.DetectNonsilenceRegion<float>(0, {in, 0, 1.f, false,
                                                                   this->window_size_, -1});
  ASSERT_EQ(nonsilence_region, nonsilence_region_);
}


TEST_F(NonsilenceOpTest, LeadTrailThreshTest) {
  std::vector<float> t0 = {0, 0, 0, 0, 0, 1.5, -100, 1.5};
  EXPECT_EQ(NonsilenceOperatorCpu::Impl::LeadTrailThresh(make_cspan(t0), .5f),
            std::make_pair(5, 3));

  std::vector<float> t1 = {1.5, -100, 1.5, 0, 0, 0, 0};
  EXPECT_EQ(NonsilenceOperatorCpu::Impl::LeadTrailThresh(make_cspan(t1), .5f),
            std::make_pair(0, 3));

  std::vector<float> t2 = {0, 0, 0, 0, 0, 1.5, -100, -100, 1.5, 0, 0, 0, 0};
  EXPECT_EQ(NonsilenceOperatorCpu::Impl::LeadTrailThresh(make_cspan(t2), 1.5f),
            std::make_pair(5, 4));

  std::vector<int> t3 = {23, 62, 46, 12, 53};
  EXPECT_EQ(NonsilenceOperatorCpu::Impl::LeadTrailThresh(make_cspan(t3), 100).second, 0);

  std::vector<int64_t> t4 = {623, 45, 62, 46, 23};
  EXPECT_EQ(NonsilenceOperatorCpu::Impl::LeadTrailThresh(make_cspan(t4), 10L),
            std::make_pair(0, 5));
}

}  // namespace testing
}  // namespace dali
