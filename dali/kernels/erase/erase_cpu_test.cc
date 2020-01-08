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

namespace dali {
namespace kernels {
namespace test {

class EraseCpuTest : public::testing::TestWithParam<
    std::tuple<TensorShape<-1>,  // data shape
    TensorShape<-1>,  // roi anchor
    TensorShape<-1>,  // roi shape
    std::vector<float>,  // fill values
    int  // channels dim (if < 0, we will use a single fill value for all channels)
  >> {
 public:
  EraseCpuTest()
    : data_shape_(std::get<0>(GetParam()).to_static<3>())
    , roi_anchor_(std::get<1>(GetParam()).to_static<3>())
    , roi_shape_(std::get<2>(GetParam()).to_static<3>())
    , fill_values_(std::get<3>(GetParam()))
    , channels_dim_(std::get<4>(GetParam()))
    , data_(volume(data_shape_))
    , in_view_(data_.data(), data_shape_) {}

  ~EraseCpuTest() override = default;

  void RunTest();

 protected:
  void SetUp() final {
    std::mt19937 rng;
    UniformRandomFill(in_view_, rng, 0.0, 1.0);
  }
  TensorShape<3> data_shape_;
  TensorShape<3> roi_anchor_;
  TensorShape<3> roi_shape_;
  std::vector<float> fill_values_;
  int channels_dim_;
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

template <typename T, int Dims>
void VerifyErase(OutTensorCPU<T, Dims> out_view,
                 OutTensorCPU<T, Dims> in_view,
                 const TensorShape<Dims> &roi_anchor,
                 const TensorShape<Dims> &roi_shape,
                 const T* fill_values,
                 int channels_dim = -1) {
  ASSERT_EQ(out_view.shape, in_view.shape);
  const auto &data_shape = out_view.shape;
  for (int i0 = 0, k = 0; i0 < data_shape[0]; i0++) {
    int roi_end_0 = (roi_anchor[0] + roi_shape[0]);
    for (int i1 = 0; i1 < data_shape[1]; i1++) {
      int roi_end_1 = (roi_anchor[1] + roi_shape[1]);
      for (int i2 = 0; i2 < data_shape[2]; i2++, k++) {
        int roi_end_2 = (roi_anchor[2] + roi_shape[2]);
        bool erased =
             i0 >= roi_anchor[0] && i0 < roi_end_0
          && i1 >= roi_anchor[1] && i1 < roi_end_1
          && i2 >= roi_anchor[2] && i2 < roi_end_2;
        int c = -1;
        if (channels_dim == 0) {
          c = i0;
        } else if (channels_dim == 1) {
          c = i1;
        } else if (channels_dim == 2) {
          c = i2;
        }
        auto fill_value = channels_dim < 0 ? fill_values[0] : fill_values[c];
        auto expected_value = erased ? fill_value : in_view.data[k];
        EXPECT_EQ(expected_value, out_view.data[k])
          << make_string(i0, ",", i1, ",", i2, " roi_start: ",
                         roi_anchor[0], ",", roi_anchor[1], ",", roi_anchor[2], " roi_end ",
                         roi_end_0, ",", roi_end_1, ",", roi_end_2,
                         " erased ", erased);
      }
    }
  }
}

void EraseCpuTest::RunTest() {
  using T = float;
  constexpr int Dims = 3;

  auto shape = in_view_.shape;
  auto size = volume(shape);

  KernelContext ctx;
  kernels::EraseArgs<T, Dims> args;
  args.rois = {{roi_anchor_, roi_shape_}};
  if (channels_dim_ > 0 || fill_values_.size() == 1) {
    args.rois[0].channels_dim = channels_dim_;
    args.rois[0].fill_values.clear();
    for (auto v : fill_values_) {
      args.rois[0].fill_values.push_back(v);
    }
  }

  kernels::EraseCpu<T, Dims> kernel;
  auto req = kernel.Setup(ctx, in_view_, args);

  // Shape should be the same as the input
  ASSERT_EQ(shape, req.output_shapes[0][0]);
  std::vector<T> out(size, 0.0f);
  auto out_view = OutTensorCPU<T, Dims>(out.data(), shape.to_static<Dims>());

  kernel.Run(ctx, out_view, in_view_, args);

  LOG_LINE << "in:\n";
  print_data(in_view_);

  LOG_LINE << "out:\n";
  print_data(out_view);

  VerifyErase(out_view, in_view_, roi_anchor_, roi_shape_,
              fill_values_.data(), channels_dim_);
}

class EraseCpuTestMultiChannel : public EraseCpuTest {};
TEST_P(EraseCpuTestMultiChannel, EraseCpuTestMultiChannel) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(EraseCpuTestMultiChannel, EraseCpuTestMultiChannel, testing::Combine(
    testing::Values(TensorShape<>{54, 55, 3}),  // data shape
    testing::Values(TensorShape<>{1, 2, 0}),  // roi anchor
    testing::Values(TensorShape<>{2, 2, 3}),  // roi shape
    testing::Values(std::vector<float>{0.0, 200.0, 300.0}),  // fill values
    testing::Values(-1, 2)));  // channels dim


class EraseCpuTestSingleChannel : public EraseCpuTest {};
TEST_P(EraseCpuTestSingleChannel, EraseCpuTestSingleChannel) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(EraseCpuTestSingleChannel, EraseCpuTestSingleChannel, testing::Combine(
    testing::Values(TensorShape<>{15, 33, 1}),  // data shape
    testing::Values(TensorShape<>{1, 2, 0}),  // roi anchor
    testing::Values(TensorShape<>{2, 2, 1}),  // roi shape
    testing::Values(std::vector<float>{100.99}),  // fill values
    testing::Values(-1, 2)));  // channels dim

}  // namespace test
}  // namespace kernels
}  // namespace dali
