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

#include "dali/kernels/signal/dct/dct_cpu.h"
#include <gtest/gtest.h>
#include <complex>
#include <tuple>
#include <vector>
#include "dali/kernels/scratch.h"
#include "dali/kernels/common/utils.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "dali/kernels/signal/dct/dct_test.h"

namespace dali {
namespace kernels {
namespace signal {
namespace dct {
namespace test {

class Dct1DCpuTest : public ::testing::TestWithParam<
  std::tuple<std::array<int64_t, 2>, int, int, bool, int>> {
 public:
  Dct1DCpuTest()
      : data_shape_(std::get<0>(GetParam()))
      , dct_type_(std::get<1>(GetParam()))
      , axis_(std::get<2>(GetParam()))
      , normalize_(std::get<3>(GetParam()))
      , ndct_(std::get<4>(GetParam()))
      , data_(volume(data_shape_))
      , in_view_(data_.data(), data_shape_) {}

  ~Dct1DCpuTest() override = default;

 protected:
  void SetUp() final {
    std::mt19937_64 rng;
    UniformRandomFill(in_view_, rng, 0., 1.);
  }
  TensorShape<2> data_shape_;
  int dct_type_ = 2;
  int axis_ = 1;
  bool normalize_ = false;
  int ndct_ = -1;

  std::vector<float> data_;
  OutTensorCPU<float, 2> in_view_;
};

TEST_P(Dct1DCpuTest, DctTest) {
  using OutputType = float;
  using InputType = float;
  constexpr int Dims = 2;
  Dct1DCpu<OutputType, InputType, Dims> kernel;
  check_kernel<decltype(kernel)>();

  if (normalize_ && dct_type_ == 1) {
    return;  // Unsupported, skip this test.
  }

  KernelContext ctx;

  auto in_shape = in_view_.shape;
  ASSERT_GT(volume(in_shape), 0);
  ASSERT_GE(axis_, 0);
  ASSERT_LT(axis_, in_shape.size());
  ASSERT_GE(dct_type_, 1);
  ASSERT_LE(dct_type_, 4);

  auto n = in_shape[axis_];
  LOG_LINE << "Test n=" << n << " axis=" << axis_ << " dct_type=" << dct_type_
           << " ndct=" << ndct_ << "\n";

  DctArgs args;
  args.dct_type = dct_type_;
  args.normalize = normalize_;
  args.ndct = ndct_;

  KernelRequirements reqs = kernel.Setup(ctx, in_view_, args, axis_);

  ScratchpadAllocator scratch_alloc;
  scratch_alloc.Reserve(reqs.scratch_sizes);
  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  TensorShape<> expected_out_shape = in_shape;

  if (ndct_ <= 0 || ndct_ > n) {
    ndct_ = n;
  }
  expected_out_shape[axis_] = ndct_;

  auto out_shape = reqs.output_shapes[0][0];
  ASSERT_EQ(expected_out_shape, out_shape);

  auto out_size = volume(out_shape);
  std::vector<OutputType> out_data(out_size);

  auto out_view = OutTensorCPU<OutputType, 2>(out_data.data(), out_shape.to_static<2>());
  kernel.Run(ctx, out_view, in_view_, args, axis_);

  auto in_strides = GetStrides(in_shape);
  auto out_strides = GetStrides(out_shape);

  auto in_stride = in_strides[axis_];
  auto out_stride = in_strides[axis_];

  assert(in_shape.size() == 2);  // assuming 2D for simplicity of this test
  auto other_axis = axis_ == 1 ? 0 : 1;

  auto nframes = in_shape[other_axis];
  for (int64_t j = 0; j < nframes; j++) {
    int64_t in_idx = j * in_strides[other_axis];
    int64_t out_idx = j * out_strides[other_axis];

    LOG_LINE << "Frame " << j << " / " << nframes << "\n";

    std::vector<InputType> in_buf(n, 0);
    LOG_LINE << "Input: ";
    for (int64_t i = 0; i < n; i++) {
      in_buf[i] = in_view_.data[in_idx];
      LOG_LINE << " " << in_view_.data[in_idx];
      in_idx += in_stride;
    }
    LOG_LINE << "\n";

    std::vector<OutputType> ref(ndct_, 0);
    ReferenceDct(dct_type_, make_span(ref), make_cspan(in_buf), normalize_);

    LOG_LINE << "DCT (type " << dct_type_ << "):";
    for (int k = 0; k < ndct_; k++) {
      LOG_LINE << " " << ref[k];
      EXPECT_NEAR(ref[k], out_data[out_idx], 1e-4);
      out_idx += out_stride;
    }
    LOG_LINE << "\n";
  }
}

INSTANTIATE_TEST_SUITE_P(Dct1DCpuTest, Dct1DCpuTest, testing::Combine(
    testing::Values(std::array<int64_t, 2>{8, 8},
                    std::array<int64_t, 2>{100, 80}),  // shape
    testing::Values(1, 2, 3, 4),  // dct_type
    testing::Values(0, 1),  // axis
    testing::Values(false, true),  // normalize
    testing::Values(-1, 4)  // ndct
  ));  // NOLINT

}  // namespace test
}  // namespace dct
}  // namespace signal
}  // namespace kernels
}  // namespace dali
