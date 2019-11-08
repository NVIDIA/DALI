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
#include "dali/kernels/signal/fft/fft_cpu.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace kernels {
namespace signal {
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

void CompareFfts(float *reference, int64_t reference_step,
                 float *data, int64_t data_step,
                 int64_t data_size,
                 float eps = 1e-3) {
    int64_t ref_idx = 0;
    int64_t data_idx = 0;
    for (int i = 0; i < data_size; i++) {
      auto diff = reference[ref_idx] - data[data_idx];
      auto diff_max = reference[ref_idx] * eps;
      // Error tends to be big if the numbers are big
      if (diff_max < eps)
        diff_max = eps;
      ASSERT_LE(diff, diff_max);
      ref_idx += reference_step;
      data_idx += data_step;
    }
}

void CompareFfts(std::complex<float> *reference, int64_t reference_step,
                 std::complex<float> *data, int64_t data_step,
                 int64_t data_size,
                 float eps = 1e-3) {
    data_size *= 2;
    CompareFfts(reinterpret_cast<float*>(reference), reference_step,
                reinterpret_cast<float*>(data), data_step, data_size, eps);
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

class ComplexFft1DCpuTest : public Fft1DCpuTest {};

TEST_P(ComplexFft1DCpuTest, FftTest) {
  using OutputType = std::complex<float>;
  using InputType = float;
  constexpr int Dims = 2;
  Fft1DCpu<OutputType, InputType, Dims> kernel;
  check_kernel<decltype(kernel)>();

  KernelContext ctx;
  FftArgs args;
  args.spectrum_type = spectrum_type_;
  ASSERT_EQ(FFT_SPECTRUM_COMPLEX, args.spectrum_type);

  args.transform_axis = 1;

  auto in_shape = in_view_.shape;
  auto n = in_shape[args.transform_axis];
  auto nfft = n;
  args.nfft = nfft;

  LOG_LINE << "Test n=" << n << " nfft=" << nfft << " axis=" << args.transform_axis
           << " spectrum_type=" << args.spectrum_type << std::endl;

  KernelRequirements reqs = kernel.Setup(ctx, in_view_, args);

  ScratchpadAllocator scratch_alloc;
  scratch_alloc.Reserve(reqs.scratch_sizes);
  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  TensorShape<> expected_out_shape = in_shape;
  expected_out_shape[args.transform_axis] = nfft;

  auto out_shape = reqs.output_shapes[0][0];
  ASSERT_EQ(expected_out_shape, out_shape);

  auto out_size = volume(out_shape);
  std::vector<OutputType> out_data(out_size);

  auto out_view = OutTensorCPU<OutputType, 2>(out_data.data(), out_shape.to_static<2>());
  kernel.Run(ctx, out_view, in_view_, args);

  LOG_LINE << "FFT:" << std::endl;
  for (int i = 0; i < nfft; i++) {
    LOG_LINE << "(" << out_view.data[i].real() << "," << out_view.data[i].imag() << "i)\n";
  }

  std::vector<std::complex<float>> reference_fft(nfft);
  NaiveDft(reference_fft.data(), 1, in_view_.data, 1, n, nfft, true);
  LOG_LINE << "Reference FFT:" << std::endl;
  for (int i = 0; i < nfft; i++) {
    LOG_LINE << "(" << reference_fft[i].real() << "," << reference_fft[i].imag() << "i)\n";
  }
  CompareFfts(reference_fft.data(), 1, out_view.data, 1, out_size);
}

INSTANTIATE_TEST_SUITE_P(Fft1DCpuTest, ComplexFft1DCpuTest, testing::Combine(
    testing::Values(
      FFT_SPECTRUM_COMPLEX),
    testing::Values(std::array<int64_t, 2>{1, 4},
                    std::array<int64_t, 2>{1, 10},
                    std::array<int64_t, 2>{1, 64},
                    std::array<int64_t, 2>{1, 100},
                    std::array<int64_t, 2>{1, 4096})));

class MagnitudeFft1DCpuTest : public Fft1DCpuTest {};

TEST_P(MagnitudeFft1DCpuTest, FftTest) {
  using OutputType = float;
  using InputType = float;
  constexpr int Dims = 2;
  Fft1DCpu<OutputType, InputType, Dims> kernel;
  check_kernel<decltype(kernel)>();

  KernelContext ctx;
  FftArgs args;
  args.spectrum_type = spectrum_type_;
  ASSERT_NE(FFT_SPECTRUM_COMPLEX, args.spectrum_type);

  args.transform_axis = 1;

  auto in_shape = in_view_.shape;
  auto n = in_shape[args.transform_axis];
  auto nfft = n;
  args.nfft = nfft;

  LOG_LINE << "Test n=" << n << " nfft=" << nfft << " axis=" << args.transform_axis
           << " spectrum_type=" << args.spectrum_type << std::endl;

  KernelRequirements reqs = kernel.Setup(ctx, in_view_, args);

  ScratchpadAllocator scratch_alloc;
  scratch_alloc.Reserve(reqs.scratch_sizes);
  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  TensorShape<> expected_out_shape = in_shape;
  expected_out_shape[args.transform_axis] = nfft/2+1;

  auto out_shape = reqs.output_shapes[0][0];
  ASSERT_EQ(expected_out_shape, out_shape);

  auto out_size = volume(out_shape);
  std::vector<OutputType> out_data(out_size);

  auto out_view = OutTensorCPU<OutputType, 2>(out_data.data(), out_shape.to_static<2>());
  kernel.Run(ctx, out_view, in_view_, args);

  std::vector<std::complex<float>> reference_fft(nfft);
  NaiveDft(reference_fft.data(), 1, in_view_.data, 1, n, nfft, true);

  std::vector<float> reference(out_size);
  switch (args.spectrum_type) {
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

  CompareFfts(reference.data(), 1, out_view.data, 1, out_size);
}

INSTANTIATE_TEST_SUITE_P(Fft1DCpuTest, MagnitudeFft1DCpuTest, testing::Combine(
    testing::Values(
      FFT_SPECTRUM_MAGNITUDE, FFT_SPECTRUM_POWER, FFT_SPECTRUM_LOG_POWER),
    testing::Values(std::array<int64_t, 2>{1, 4},
                    std::array<int64_t, 2>{1, 10},
                    std::array<int64_t, 2>{1, 64},
                    std::array<int64_t, 2>{1, 100},
                    std::array<int64_t, 2>{1, 4096})));

class Fft1DCpuOtherLayoutTest : public::testing::TestWithParam<
  std::tuple<FftSpectrumType, std::array<int64_t, 3>, int>> {
 public:
  Fft1DCpuOtherLayoutTest()
    : spectrum_type_(std::get<0>(GetParam()))
    , data_shape_(std::get<1>(GetParam()))
    , transform_axis_(std::get<2>(GetParam()))
    , data_(volume(data_shape_))
    , in_view_(data_.data(), data_shape_) {}

  ~Fft1DCpuOtherLayoutTest() override = default;

 protected:
  void SetUp() final {
    std::mt19937_64 rng;
    UniformRandomFill(in_view_, rng, 0., 1.);
  }
  FftSpectrumType spectrum_type_;
  TensorShape<3> data_shape_;
  int transform_axis_;
  std::vector<float> data_;
  OutTensorCPU<float, 3> in_view_;
};

class ComplexFft1DCpuOtherLayoutTest : public Fft1DCpuOtherLayoutTest {};

TEST_P(ComplexFft1DCpuOtherLayoutTest, LayoutTest) {
  using OutputType = std::complex<float>;
  using InputType = float;
  constexpr int Dims = 3;
  Fft1DCpu<OutputType, InputType, Dims> kernel;
  check_kernel<decltype(kernel)>();

  KernelContext ctx;
  FftArgs args;
  args.spectrum_type = spectrum_type_;
  args.transform_axis = transform_axis_;

  auto in_shape = in_view_.shape;
  auto n = in_shape[args.transform_axis];
  auto nfft = n;
  args.nfft = nfft;

  LOG_LINE <<
    make_string("Test n=", n, " nfft=", nfft, " axis=", args.transform_axis,
                " spectrum_type=", args.spectrum_type) << std::endl;

  KernelRequirements reqs = kernel.Setup(ctx, in_view_, args);

  ScratchpadAllocator scratch_alloc;
  scratch_alloc.Reserve(reqs.scratch_sizes);
  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  auto expected_out_shape = in_shape;
  expected_out_shape[args.transform_axis] = nfft;

  auto out_shape = reqs.output_shapes[0][0].to_static<3>();
  ASSERT_EQ(expected_out_shape, out_shape);

  auto out_size = volume(out_shape);
  std::vector<OutputType> out(out_size);
  auto out_view = OutTensorCPU<OutputType, 3>(out.data(), out_shape);
  auto *in_data = in_view_.data;
  auto *out_data = out_view.data;
  kernel.Run(ctx, out_view, in_view_, args);

  TensorShape<> in_strides = in_shape;
  in_strides[in_strides.size()-1] = 1;
  for (int d = in_strides.size()-2; d >= 0; d--) {
    in_strides[d] = in_strides[d+1] * in_shape[d+1];
  }
  int64_t in_step = in_shape[args.transform_axis];
  int64_t in_stride = in_strides[args.transform_axis];

  TensorShape<> out_strides = out_shape;
  out_strides[out_strides.size()-1] = 1;
  for (int d = out_strides.size()-2; d >= 0; d--) {
    out_strides[d] = out_strides[d+1] * out_shape[d+1];
  }
  int64_t out_stride = out_strides[args.transform_axis];

  std::vector<std::complex<float>> reference_fft(nfft);

  int64_t total_ffts = volume(in_shape) / n;
  LOG_LINE << "in_shape " << in_shape << std::endl;
  LOG_LINE << "in_strides " << in_strides << std::endl;

  LOG_LINE << "in : \n[" << std::endl;;
  for (int i0 = 0; i0 < in_shape[0]; i0++) {
    LOG_LINE << "\t[" << std::endl;
    for (int i1 = 0; i1 < in_shape[1]; i1++) {
      LOG_LINE << "\t\t[";
      for (int i2 = 0; i2 < in_shape[2]; i2++) {
        int idx = i0*in_strides[0]+i1*in_strides[1]+i2*in_strides[2];
        ASSERT_LT(idx, volume(in_shape));
        LOG_LINE << " " << in_data[idx];
      }
      LOG_LINE << " ]" << std::endl;
    }
    LOG_LINE << "\t]" << std::endl;
  }
  LOG_LINE << "]" << std::endl;


  LOG_LINE << "out : \n[" << std::endl;;
  for (int i0 = 0; i0 < out_shape[0]; i0++) {
    LOG_LINE << "\t[" << std::endl;
    for (int i1 = 0; i1 < out_shape[1]; i1++) {
      LOG_LINE << "\t\t[";
      for (int i2 = 0; i2 < out_shape[2]; i2++) {
        int idx = i0*out_strides[0]+i1*out_strides[1]+i2*out_strides[2];
        ASSERT_LT(idx, volume(out_shape));
        LOG_LINE << " " << out_data[idx];
      }
      LOG_LINE << " ]" << std::endl;
    }
    LOG_LINE << "\t]" << std::endl;
  }
  LOG_LINE << "]" << std::endl;

  LOG_LINE << "n " << n << std::endl;
  LOG_LINE << "nfft " << nfft << std::endl;
  LOG_LINE << "axis " << transform_axis_ << std::endl;

  LOG_LINE << "in_shape " << in_shape << std::endl;
  LOG_LINE << "out_shape " << out_shape << std::endl;

  LOG_LINE << "in_strides " << in_strides << std::endl;
  LOG_LINE << "out_strides " << out_strides << std::endl;

  LOG_LINE << "in_stride " << in_stride << std::endl;
  LOG_LINE << "out_stride " << out_stride << std::endl;

  std::vector<int> dims;
  for (int d = 0; d < 3; d++) {
    if (d != transform_axis_)
      dims.push_back(d);
  }

  std::vector<float> in_data_buf(n, 0);
  auto *in_data_ptr = in_view_.data;
  std::vector<OutputType> out_data_buf(nfft, {0.0f, 0.0f});
  auto *out_data_ptr = out_view.data;

  for (int i = 0; i < in_shape[dims[0]]; i++) {
    auto* in_data_ptr0 = in_view_.data + i * in_strides[dims[0]];
    auto* out_data_ptr0 = out_view.data + i * out_strides[dims[0]];
    for (int j = 0; j < in_shape[dims[1]]; j++) {
      auto* in_data_ptr1 = in_data_ptr0 + j * in_strides[dims[1]];
      auto* out_data_ptr1 = out_data_ptr0 + j * out_strides[dims[1]];
      LOG_LINE << "In data " << i << "," << j << " : [";
      for (int k = 0; k < n; k++) {
        in_data_buf[k] = in_data_ptr1[k*in_stride];
        LOG_LINE << " " << in_data_buf[k];
      }
      LOG_LINE << " ]\n";

      for (int k = 0; k < nfft; k++) {
        out_data_buf[k] = out_data_ptr1[k*out_stride];
      }

      NaiveDft(reference_fft.data(), 1, in_data_buf.data(), 1, n, nfft, true);

      LOG_LINE << "Reference data: ";
      for (int k = 0; k < nfft; k++) {
        LOG_LINE << " (" << reference_fft[k].real() << ", " << reference_fft[k].imag() << "),";
      }
      LOG_LINE << std::endl;

      LOG_LINE << "Actual data: ";
      for (int k = 0; k < nfft; k++) {
        LOG_LINE << " (" << out_data_buf[k].real() << ", " << out_data_buf[k].imag() << "),";
      }
      LOG_LINE << std::endl;

      CompareFfts(reference_fft.data(), 1, out_data_buf.data(), 1, nfft);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(ComplexFft1DCpuOtherLayoutTest, ComplexFft1DCpuOtherLayoutTest,
  testing::Combine(
    testing::Values(FFT_SPECTRUM_COMPLEX),
    testing::Values(std::array<int64_t, 3>{6, 8, 4}),
    testing::Values(0, 1, 2)));



}  // namespace test
}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali
