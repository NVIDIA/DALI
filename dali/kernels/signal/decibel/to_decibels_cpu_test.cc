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
#include "dali/kernels/signal/decibel/to_decibels_cpu.h"
#include "dali/kernels/common/utils.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace kernels {
namespace signal {
namespace test {

class ToDecibelsCpuTest : public::testing::TestWithParam<
  std::tuple<std::array<int64_t, 2> /* data_shape */,
             float /* mul */,
             float /* s_ref */,
             float /* min_ratio */,
             float /* data_max */,
             bool  /* ref_max */>> {
 public:
  ToDecibelsCpuTest()
    : data_shape_(std::get<0>(GetParam()))
    , mul_(std::get<1>(GetParam()))
    , s_ref_(std::get<2>(GetParam()))
    , min_ratio_(std::get<3>(GetParam()))
    , data_max_(std::get<4>(GetParam()))
    , ref_max_(std::get<5>(GetParam()))
    , data_(volume(data_shape_))
    , in_view_(data_.data(), data_shape_) {}

  ~ToDecibelsCpuTest() override = default;

 protected:
  void SetUp() final {
    std::mt19937 rng;
    UniformRandomFill(in_view_, rng, 0.0, data_max_);
  }
  TensorShape<> data_shape_;
  float mul_ = 10.0;
  float s_ref_ = 1.0;
  float min_ratio_ = 1e-8;
  float data_max_ = 1.0;
  bool ref_max_ = false;
  std::vector<float> data_;
  OutTensorCPU<float, DynamicDimensions> in_view_;
};

template <typename T>
void print_data(const OutTensorCPU<T, DynamicDimensions>& data_view) {
  auto sh = data_view.shape;
  for (int i0 = 0; i0 < sh[0]; i0++) {
    for (int i1 = 0; i1 < sh[1]; i1++) {
      int k = i0 * sh[1] + i1;
      LOG_LINE << " " << data_view.data[k];
    }
    LOG_LINE << "\n";
  }
}

TEST_P(ToDecibelsCpuTest, ToDecibelsCpuTest) {
  using T = float;
  ToDecibelsCpu<T> kernel;
  check_kernel<decltype(kernel)>();

  KernelContext ctx;
  ToDecibelsArgs<T> args;
  args.multiplier = mul_;
  args.s_ref = s_ref_;
  args.min_ratio = min_ratio_;
  args.ref_max = ref_max_;

  KernelRequirements reqs = kernel.Setup(ctx, in_view_, args);
  auto out_shape = reqs.output_shapes[0][0];
  ASSERT_EQ(out_shape, in_view_.shape);

  ScratchpadAllocator scratch_alloc;
  scratch_alloc.Reserve(reqs.scratch_sizes);
  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  auto out_size = volume(out_shape);
  std::vector<T> expected_out(out_size);
  auto expected_out_view =
      OutTensorCPU<T, DynamicDimensions>(expected_out.data(), out_shape);

  if (ref_max_) {
    s_ref_ = 0.0;
    for (int64_t i = 0; i < out_size; i++) {
      if (in_view_.data[i] > s_ref_)
        s_ref_ = in_view_.data[i];
    }
  }

  for (int64_t i = 0; i < out_size; i++) {
    expected_out_view.data[i] =
      args.multiplier * std::log10(std::max(args.min_ratio, in_view_.data[i] / s_ref_));
  }

  LOG_LINE << "in:\n";
  print_data(in_view_);

  LOG_LINE << "expected out:\n";
  print_data(expected_out_view);

  LOG_LINE << "data max: " << data_max_ << std::endl;
  LOG_LINE << "ref_max: " << s_ref_ << std::endl;

  std::vector<T> out(out_size);
  auto out_view = OutTensorCPU<T, DynamicDimensions>(out.data(), out_shape);
  kernel.Run(ctx, out_view, in_view_, args);

  LOG_LINE << "out:\n";
  print_data(out_view);

  for (int idx = 0; idx < volume(out_view.shape); idx++) {
    ASSERT_NEAR(expected_out[idx], out_view.data[idx], 1e-4) <<
      "Output data doesn't match reference (idx=" << idx << ")";
  }
}

INSTANTIATE_TEST_SUITE_P(ToDecibelsCpuTest, ToDecibelsCpuTest, testing::Combine(
    testing::Values(std::array<int64_t, 2>{1, 100},
                    std::array<int64_t, 2>{2, 12}),
    testing::Values(10.0, 20.0),     // mul
    testing::Values(1.0, 1e-6),      // s_ref
    testing::Values(1e-8, 1e-20),    // min_ratio
    testing::Values(1.0, 10000.0),   // data_max
    testing::Values(true, false)));  // ref_max

}  // namespace test
}  // namespace signal
}  // namespace kernels
}  // namespace dali
