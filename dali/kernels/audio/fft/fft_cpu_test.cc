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
#include "dali/kernels/audio/fft/fft_cpu.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace kernels {
namespace audio {
namespace fft {
namespace test {

void NaiveDft(std::complex<float> *out,
              int64_t out_stride,
              float *in,
              int64_t in_stride,
              int64_t n,
              int64_t nfft,
              bool full_spectrum = true) {
  // Loop through each sample in the frequency domain
  for (int64_t k = 0; k < (full_spectrum ? nfft : (nfft/2+1)); k++) {
    float real = 0.0f, imag = 0.0f;
    // Loop through each sample in the time domain
    for (int64_t i = 0; i < n; i++) {
      auto x = in[i * in_stride];
      real += x * cos(2.0f * M_PI * k * i / n);
      imag += -x * sin(2.0f * M_PI * k * i / n);
    }
    out[k * out_stride] = {real, imag};
  }
}

void PowerSpectrum(float *out,
                   int64_t out_stride,
                   std::complex<float> *in,  // fft
                   int64_t in_stride,
                   int64_t nfft) {
  for (int64_t k = 0; k <= nfft / 2; k++) {
    auto real = in[k * in_stride].real();
    auto imag = in[k * in_stride].imag();
    out[k * out_stride] = real*real + imag*imag;
  }
}

void MagSpectrum(float *out,
                 int64_t out_stride,
                 std::complex<float> *in,  // fft
                 int64_t in_stride,
                 int64_t nfft) {
  for (int64_t k = 0; k <= nfft / 2; k++) {
    auto real = in[k * in_stride].real();
    auto imag = in[k * in_stride].imag();
    out[k * out_stride] = sqrt(real*real + imag*imag);
  }
}

void LogPowerSpectrum(float *out,
                      int64_t out_stride,
                      std::complex<float> *in,  // fft
                      int64_t in_stride,
                      int64_t nfft) {
  for (int64_t k = 0; k <= nfft / 2; k++) {
    auto real = in[k * in_stride].real();
    auto imag = in[k * in_stride].imag();
    out[k * out_stride] = 10 * log10(real*real + imag*imag);
  }
}

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

  auto in_shape = in_view_.shape;
  auto n = in_shape[args.transform_axis];
  auto nfft = n;
  args.nfft = nfft;

  KernelRequirements reqs = kernel.Setup(ctx, in_view_, args);

  ScratchpadAllocator scratch_alloc;
  scratch_alloc.Reserve(reqs.scratch_sizes);
  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  auto expected_out_shape = in_shape;
  expected_out_shape[args.transform_axis] = args.spectrum_type == FFT_SPECTRUM_COMPLEX ?
    nfft * 2 : (nfft/2+1);

  auto out_shape = reqs.output_shapes[0][0].to_static<2>();
  ASSERT_EQ(expected_out_shape, out_shape);

  auto out_size = volume(out_shape);
  std::vector<float> out_data(out_size);

  auto out_view = OutTensorCPU<float, 2>(out_data.data(), out_shape);
  kernel.Run(ctx, out_view, in_view_, args);


  if (args.spectrum_type == FFT_SPECTRUM_COMPLEX) {
    LOG_LINE << "FFT:" << std::endl;
    for (int i = 0; i < nfft; i++) {
      LOG_LINE << "(" << out_view.data[2*i] << "," << out_view.data[2*i+1] << "i)" << std::endl;
    }
  }

  std::vector<std::complex<float>> reference_fft(2*nfft);
  NaiveDft(reference_fft.data(), 1, in_view_.data, 1, n, nfft, true);

  std::vector<float> reference(out_size);
  switch (args.spectrum_type) {
    case FFT_SPECTRUM_COMPLEX:
      memcpy(reference.data(), reference_fft.data(), reference_fft.size()*sizeof(float));
      break;
    case FFT_SPECTRUM_POWER:
      PowerSpectrum(reference.data(), 1, reference_fft.data(), 1, nfft);
      break;
    case FFT_SPECTRUM_MAGNITUDE:
      MagSpectrum(reference.data(), 1, reference_fft.data(), 1, nfft);
      break;
    case FFT_SPECTRUM_LOG_POWER:
      LogPowerSpectrum(reference.data(), 1, reference_fft.data(), 1, nfft);
      break;
    default:
      ASSERT_TRUE(false);
  }

  if (args.spectrum_type == FFT_SPECTRUM_COMPLEX) {
    LOG_LINE << "Reference FFT:" << std::endl;
    for (int i = 0; i < nfft; i++) {
      LOG_LINE << "(" << reference[2*i] << "," << reference[2*i+1] << "i)" << std::endl;
    }
  }

  float eps = 1e-3;
  for (int i = 0; i < out_size; i++) {
    auto diff = reference[i] - out_view.data[i];
    auto diff_max = reference[i] * eps;
    // Error tends to be big if the numbers are big
    if (diff_max < eps)
      diff_max = eps;
    EXPECT_LE(diff, diff_max);
  }
}

INSTANTIATE_TEST_SUITE_P(Fft1DCpuTest, Fft1DCpuTest, testing::Combine(
    testing::Values(
      FFT_SPECTRUM_COMPLEX, FFT_SPECTRUM_MAGNITUDE, FFT_SPECTRUM_POWER, FFT_SPECTRUM_LOG_POWER),
    testing::Values(std::array<int64_t, 2>{1, 4},
                    std::array<int64_t, 2>{1, 10},
                    std::array<int64_t, 2>{1, 64},
                    std::array<int64_t, 2>{1, 100},
                    std::array<int64_t, 2>{1, 4096})));

}  // namespace test
}  // namespace fft
}  // namespace audio
}  // namespace kernels
}  // namespace dali
