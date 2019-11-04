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
  std::tuple<FftSpectrumType, std::array<int64_t, 2>>> {
 public:
  Fft1DCpuTest()
    : spectrum_type_(std::get<0>(GetParam()))
    , data_shape_(std::get<1>(GetParam()))
    , data_(volume(data_shape_))
    , in_view_(data_.data(), data_shape_) {}

  ~Fft1DCpuTest() override = default;

 protected:
  void SetUp() final {
    std::mt19937_64 rng;
    UniformRandomFill(in_view_, rng, 0., 1.);
  }
  FftSpectrumType spectrum_type_;
  TensorShape<2> data_shape_;
  std::vector<float> data_;
  OutTensorCPU<float, 2> in_view_;
};

TEST_P(Fft1DCpuTest, KernelTest) {
  KernelContext ctx;
  Fft1DCpu<float> kernel;
  FftArgs args;
  args.spectrum_type = spectrum_type_;
  args.transform_axis = 1;
  KernelRequirements reqs = kernel.Setup(ctx, in_view_, args);

  ScratchpadAllocator scratch_alloc;
  scratch_alloc.Reserve(reqs.scratch_sizes);
  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  auto in_shape = in_view_.shape;
  auto n = in_shape[in_shape.size()-1];

  auto expected_out_shape = in_shape;
  int64_t nfft = 2;
  while (nfft < n)
    nfft *= 2;
  expected_out_shape[1] = args.spectrum_type == FFT_SPECTRUM_COMPLEX ?
    nfft * 2 : (nfft/2+1);

  auto out_shape = reqs.output_shapes[0][0].to_static<2>();
  ASSERT_EQ(expected_out_shape, out_shape);

  std::vector<float> out_data(volume(out_shape));

  auto out_view = OutTensorCPU<float, 2>(out_data.data(), out_shape);
  kernel.Run(ctx, out_view, in_view_, args);

  LOG_LINE << "FFT:" << std::endl;
  for (int i = 0; i < nfft; i++) {
    LOG_LINE << "(" << out_view.data[2*i] << "," << out_view.data[2*i+1] << "i)" << std::endl;
  }

  // Asserting that the right side of the spectrum is mirrored as expected
  if (args.spectrum_type == FFT_SPECTRUM_COMPLEX) {
    for (int i = nfft / 2 + 1; i < nfft; i++) {
      ASSERT_NEAR(out_view.data[2*i+0],  out_view.data[2*(nfft-i)+0], 1e-5);
      ASSERT_NEAR(out_view.data[2*i+1], -out_view.data[2*(nfft-i)+1], 1e-5);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Fft1DCpuTest, Fft1DCpuTest, testing::Combine(
    testing::Values(
      FFT_SPECTRUM_COMPLEX, FFT_SPECTRUM_MAGNITUDE, FFT_SPECTRUM_POWER, FFT_SPECTRUM_LOG_POWER),
    testing::Values(std::array<int64_t, 2>{1, 64},
                    std::array<int64_t, 2>{1, 4096},
                    std::array<int64_t, 2>{1, 100})));

}  // namespace test
}  // namespace fft
}  // namespace audio
}  // namespace kernels
}  // namespace dali
