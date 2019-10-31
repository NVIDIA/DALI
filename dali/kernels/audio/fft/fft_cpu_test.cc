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
#include "dali/kernels/scratch.h"
#include "dali/kernels/audio/fft/fft_cpu.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace kernels {
namespace audio {
namespace fft {
namespace test {

class Fft1DCpuTest : public::testing::TestWithParam<
  std::tuple<FftOutputType, std::array<int64_t, 2>>> {
 public:
  Fft1DCpuTest()
    : output_type_(std::get<0>(GetParam()))
    , data_shape_(std::get<1>(GetParam()))
    , data_(volume(data_shape_))
    , in_view_(data_.data(), data_shape_) {}

  ~Fft1DCpuTest() override = default;

 protected:
  void SetUp() final {
    std::mt19937_64 rng;
    UniformRandomFill(in_view_, rng, 0., 1.);
  }
  FftOutputType output_type_;
  TensorShape<2> data_shape_;
  std::vector<float> data_;
  OutTensorCPU<float, 2> in_view_;
};

TEST_P(Fft1DCpuTest, KernelTest) {
  KernelContext ctx;
  Fft1DCpu<float> kernel;
  FftArgs args = {output_type_, 0, 1};
  KernelRequirements reqs = kernel.Setup(ctx, in_view_, args);

  ScratchpadAllocator scratch_alloc;
  scratch_alloc.Reserve(reqs.scratch_sizes);
  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  auto out_shape = reqs.output_shapes[0][0].to_static<2>();
  std::vector<float> out_data(volume(out_shape));

  auto out_view = OutTensorCPU<float, 2>(out_data.data(), out_shape);
  kernel.Run(ctx, out_view, in_view_, args);

  auto in_shape = in_view_.shape;
  auto in_nchannels = in_shape[in_shape.size()-1];

  auto expected_out_shape = in_shape;
  ASSERT_TRUE(args.output_type == FFT_OUTPUT_TYPE_MAGNITUDE ||
              args.output_type == FFT_OUTPUT_TYPE_COMPLEX);
  if (args.output_type == FFT_OUTPUT_TYPE_COMPLEX) {
    expected_out_shape[out_shape.size()-1] *= 2;
  }
  ASSERT_EQ(expected_out_shape, out_view.shape);

  auto n = in_shape[0];

  LOG_LINE << "FFT:" << std::endl;
  for (int i = 0; i < n; i++) {
    LOG_LINE << "(" << out_view.data[2*i] << "," << out_view.data[2*i+1] << "i)" << std::endl;
  }

  // Asserting that the right side of the spectrum is mirrored as expected
  for (int i = n/2+1; i < n; i++) {
    ASSERT_NEAR(out_view.data[2*i+0],  out_view.data[2*(n-i)+0], 1e-5);
    ASSERT_NEAR(out_view.data[2*i+1], -out_view.data[2*(n-i)+1], 1e-5);
  }
}

INSTANTIATE_TEST_SUITE_P(Fft1DCpuTest, Fft1DCpuTest, testing::Combine(
    testing::Values(FFT_OUTPUT_TYPE_COMPLEX),
    testing::Values(std::array<int64_t, 2>{64, 1},
                    std::array<int64_t, 2>{4096, 1},
                    std::array<int64_t, 2>{100, 1})));

}  // namespace test
}  // namespace fft
}  // namespace audio
}  // namespace kernels
}  // namespace dali
