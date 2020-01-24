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
#include <utility>
#include <random>
#include "dali/kernels/signal/moving_mean_square.h"
#include "dali/operators/audio/nonsilence_op.h"

namespace dali {
namespace testing {

class NonsilenceOpTest : public ::testing::Test {
 public:
  NonsilenceOpTest() {
//    input_.resize(shape_.num_elements());
  }


  void SetUp() final {
    FillInputAndReference();
  }


  std::vector<float> input_;
  std::pair<int, int> nonsilence_region_;
//  std::vector<float> ref_output_;
//  int window_size_ = 2048;
  int buffer_length_ = 10;
//  int reset_interval_ = 5001;
  TensorShape<1> shape_ = {buffer_length_};
//  TensorShape<kNDims> out_shape_ = {buffer_length_ - window_size_ + 1};

 private:

  void FillInputAndReference() {
    const int nrolls = 10000;
    nonsilence_region_ = {4, 2};
    std::default_random_engine generator{42};
    std::normal_distribution<double> distribution(5.0, .1);
    input_.resize(buffer_length_);
    for (int i = 0; i < nrolls; ++i) {
      double number = distribution(generator);
      if ((number >= 0) && (number < buffer_length_)) input_[int(number)]++;
    }
  }
//  void calc_output() {
//    ref_output_.resize(buffer_length_ - window_size_);
//    for (int i = 0; i <= buffer_length_ - window_size_; i++) {
//      float sumsq = 0;
//      for (int j = 0; j < window_size_; j++) {
//        auto val = static_cast<float>(input_[i + j]);
//        sumsq += val * val;
//      }
//      ref_output_[i] = sumsq / window_size_;
//    }
//  }


//  template<typename RNG, typename T = InputType>
//  std::enable_if_t<std::is_signed<T>::value>
//  FillInput(RNG &rng) {
//    UniformRandomFill(input_, rng, -100, 100);
//  }


//  template<typename RNG, typename T = InputType>
//  std::enable_if_t<!std::is_signed<T>::value>
//  FillInput(RNG &rng) {
//    UniformRandomFill(input_, rng, 0, 100);
//  }
};

using TestedKernel = kernels::signal::MovingMeanSquareCpu<float>;

TEST_F(NonsilenceOpTest, KernelTest) {
  NonsilenceOperatorCpuImpl ns_op;
  ns_op.SetupKernel<TestedKernel>(1, 1);
  auto in = make_tensor_cpu(reinterpret_cast<const float*>(this->input_.data()), this->shape_);
  ns_op.RunKernel<float, TestedKernel>(in, 0, 0);
}

//TEST(NonsilenceOpTest, DetectNonsilenceRegion) {
//  std::vector<float> t0 = {0, 0, 0, 0, 0, 1.5, -100, 1.5};
//  EXPECT_EQ(detail::DetectNonsilenceRegion(make_cspan(t0), .5f), std::make_pair(5, 3));
//
//  std::vector<float> t1 = {1.5, -100, 1.5, 0, 0, 0, 0};
//  EXPECT_EQ(detail::DetectNonsilenceRegion(make_cspan(t1), .5f), std::make_pair(0, 3));
//
//  std::vector<float> t2 = {0, 0, 0, 0, 0, 1.5, -100, -100, 1.5, 0, 0, 0, 0};
//  EXPECT_EQ(detail::DetectNonsilenceRegion(make_cspan(t2), 1.5f), std::make_pair(5, 4));
//
//  std::vector<int> t3 = {23, 62, 46, 12, 53};
//  EXPECT_EQ(detail::DetectNonsilenceRegion(make_cspan(t3), 100).second, 0);
//
//  std::vector<int64_t> t4 = {623, 45, 62, 46, 23};
//  EXPECT_EQ(detail::DetectNonsilenceRegion(make_cspan(t4), 10L), std::make_pair(0, 5));
//}

}  // namespace testing
}  // namespace dali
