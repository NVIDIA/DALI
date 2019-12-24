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
#include <tuple>
#include <vector>
#include <complex>
#include <cmath>
#include "dali/kernels/scratch.h"
#include "dali/kernels/erase/erase_cpu.h"
#include "dali/kernels/common/utils.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"

#undef LOG_LINE
#define LOG_LINE std::cout

namespace dali {
namespace kernels {
namespace test {

class EraseCpuTest : public::testing::TestWithParam<
  std::tuple<std::array<int64_t, 3>,  /* data_shape */ 
  std::array<int64_t, 3>, 
  std::array<int64_t, 3>>> { 
 public:
  EraseCpuTest()
    : data_shape_(std::get<0>(GetParam()))
    , roi_anchor_(std::get<1>(GetParam()))
    , roi_shape_(std::get<2>(GetParam()))
    , data_(volume(data_shape_))
    , in_view_(data_.data(), data_shape_) {}

  ~EraseCpuTest() override = default;

 protected:
  void SetUp() final {
    std::mt19937 rng;
    UniformRandomFill(in_view_, rng, 0.0, 1.0);
  }
  TensorShape<3> data_shape_;
  TensorShape<3> roi_anchor_;
  TensorShape<3> roi_shape_;
  std::vector<float> data_;
  OutTensorCPU<float, 3> in_view_;
};


template <typename T>
void print_data(const OutTensorCPU<T, 3>& data_view) {
  auto sh = data_view.shape;
  int k = 0;
  for (int i0 = 0; i0 < sh[0]; i0++) {
    for (int i1 = 0; i1 < sh[1]; i1++) {
      for (int i2 = 0; i2 < sh[2]; i2++) {
        LOG_LINE << " " << data_view.data[k++];
      }
      if (sh[2] > 1) LOG_LINE << "\n";
    }
    LOG_LINE << "\n";
  }
  LOG_LINE << "\n";  
}

TEST_P(EraseCpuTest, EraseCpuTest) {
  using T = float;
  constexpr int Dims = 3;

  auto shape = in_view_.shape;
  auto size = volume(shape);

  KernelContext ctx;
  kernels::EraseArgs<Dims> args;
  args.rois = {{roi_anchor_, roi_shape_}};

  kernels::EraseCpu<T, Dims> kernel;
  auto req = kernel.Setup(ctx, in_view_, args);

  // Shape should be the same as the inputout_
  ASSERT_EQ(shape, req.output_shapes[0][0]);
  std::vector<T> out(size, 0.0f);
  auto out_view = OutTensorCPU<T, Dims>(out.data(), shape.to_static<Dims>());

  kernel.Run(ctx, out_view, in_view_, args);

  LOG_LINE << "in:\n";
  print_data(in_view_);

  LOG_LINE << "out:\n";
  print_data(out_view);

  for (int i0 = 0, k = 0; i0 < data_shape_[0]; i0++) {
    int roi_end_0 = (roi_anchor_[0] + roi_shape_[0]);
    for (int i1 = 0; i1 < data_shape_[1]; i1++) {
      int roi_end_1 = (roi_anchor_[1] + roi_shape_[1]);
      for (int i2 = 0; i2 < data_shape_[2]; i2++, k++) {
        int roi_end_2 = (roi_anchor_[2] + roi_shape_[2]);
        bool erased = 
             i0 >= roi_anchor_[0] && i0 < roi_end_0 
          && i1 >= roi_anchor_[1] && i1 < roi_end_1 
          && i2 >= roi_anchor_[2] && i2 < roi_end_2;
        auto expected_value = erased ? T(0) : in_view_.data[k];
        EXPECT_EQ(expected_value, out_view.data[k]) 
          << make_string(i0, ",", i1, ",", i2, " roi_start: ", 
                         roi_anchor_[0], ",", roi_anchor_[1], ",", roi_anchor_[2], " roi_end ",
                         roi_end_0, ",", roi_end_1, ",", roi_end_2, 
                         " erased ", erased);  
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(EraseCpuTest, EraseCpuTest, testing::Combine(
    testing::Values(std::array<int64_t, 3>{4, 4, 1}),  // data shape
    testing::Values(std::array<int64_t, 3>{1, 2, 0}),  // roi anchor
    testing::Values(std::array<int64_t, 3>{2, 2, 1})));  // roi shape

}  // namespace test
}  // namespace kernels
}  // namespace dali
