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

void NaiveDft(std::vector<std::complex<float>> &out,
              const span<const float>& in,
              int64_t nfft,
              bool full_spectrum = false) {
  auto n = in.size();
  auto out_size = full_spectrum ? nfft : nfft/2+1;
  out.clear();
  out.reserve(out_size);
  // Loop through each sample in the frequency domain
  for (int64_t k = 0; k < out_size; k++) {
    float real = 0.0f, imag = 0.0f;
    // Loop through each sample in the time domain
    for (int64_t i = 0; i < n; i++) {
      auto x = in[i];
      real += x * cos(2.0f * M_PI * k * i / n);
      imag += -x * sin(2.0f * M_PI * k * i / n);
    }
    out.push_back({real, imag});
  }
}

void CompareFfts(const span<const float>& reference,
                 const span<const float>& fft,
                 float eps = 1e-3) {
  ASSERT_EQ(reference.size(), fft.size());
  const auto *ref = reference.data(), *data = fft.data();
  for (int i = 0; i < fft.size(); i++) {
    auto diff = ref[i] - data[i];
    auto diff_max = ref[i] * eps;
    // Error tends to be big if the numbers are big
    if (diff_max < eps)
      diff_max = eps;
    ASSERT_LE(diff, diff_max);
  }
}

void CompareFfts(const span<const std::complex<float>>& reference,
                 const span<const std::complex<float>>& fft,
                 float eps = 1e-3) {
  span<const float> ref_f = {reinterpret_cast<const float*>(reference.data()), reference.size()*2};
  span<const float> fft_f = {reinterpret_cast<const float*>(fft.data()), fft.size()*2};
  CompareFfts(ref_f, fft_f, eps);
}

void PowerSpectrum(span<float> out,
                   span<std::complex<float>> in,
                   int64_t nfft) {
  for (int64_t k = 0; k <= nfft / 2; k++) {
    out[k] = std::norm(in[k]);
  }
}

void MagSpectrum(span<float> out,
                 span<std::complex<float>> in,
                 int64_t nfft) {
  for (int64_t k = 0; k <= nfft / 2; k++) {
    out[k] = std::abs(in[k]);
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
  expected_out_shape[args.transform_axis] = nfft/2+1;

  auto out_shape = reqs.output_shapes[0][0];
  ASSERT_EQ(expected_out_shape, out_shape);

  auto out_size = volume(out_shape);
  std::vector<OutputType> out_data(out_size);

  auto out_view = OutTensorCPU<OutputType, 2>(out_data.data(), out_shape.to_static<2>());
  kernel.Run(ctx, out_view, in_view_, args);

  LOG_LINE << "FFT:" << std::endl;
  for (int i = 0; i < nfft/2+1; i++) {
    LOG_LINE << "(" << out_view.data[i].real() << "," << out_view.data[i].imag() << "i)\n";
  }

  std::vector<std::complex<float>> reference_fft(nfft/2+1);
  NaiveDft(reference_fft, make_cspan(in_view_.data, n), nfft, true);
  LOG_LINE << "Reference FFT:" << std::endl;
  for (int i = 0; i < nfft/2+1; i++) {
    LOG_LINE << "(" << reference_fft[i].real() << "," << reference_fft[i].imag() << "i)\n";
  }
  CompareFfts(make_cspan(reference_fft.data(), nfft/2+1), make_cspan(out_view.data, nfft/2+1));
}

INSTANTIATE_TEST_SUITE_P(Fft1DCpuTest, ComplexFft1DCpuTest, testing::Combine(
    testing::Values(
      FFT_SPECTRUM_COMPLEX),
    testing::Values(std::array<int64_t, 2>{1, 4},
                    std::array<int64_t, 2>{1, 10},
                    std::array<int64_t, 2>{1, 64},
                    std::array<int64_t, 2>{1, 100},
                    std::array<int64_t, 2>{1, 4096})));

class PowerSpectrum1DCpuTest : public Fft1DCpuTest {};

TEST_P(PowerSpectrum1DCpuTest, FftTest) {
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

  std::vector<std::complex<float>> reference_fft(nfft/2+1);
  NaiveDft(reference_fft, make_cspan(in_view_.data, n), nfft);

  std::vector<float> reference(nfft/2+1);
  switch (args.spectrum_type) {
    case FFT_SPECTRUM_POWER:
      PowerSpectrum(make_span(reference), make_span(reference_fft), nfft);
      break;
    case FFT_SPECTRUM_MAGNITUDE:
      MagSpectrum(make_span(reference), make_span(reference_fft), nfft);
      break;
    default:
      ASSERT_TRUE(false);
  }

  CompareFfts(make_cspan(reference), make_cspan(out_view.data, out_size));
}

INSTANTIATE_TEST_SUITE_P(Fft1DCpuTest, PowerSpectrum1DCpuTest, testing::Combine(
    testing::Values(
      FFT_SPECTRUM_MAGNITUDE, FFT_SPECTRUM_POWER),
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
  expected_out_shape[args.transform_axis] = nfft/2+1;

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

  std::vector<std::complex<float>> reference_fft(nfft/2+1);

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
  std::vector<OutputType> out_data_buf(nfft/2+1, {0.0f, 0.0f});
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

      for (int k = 0; k < nfft/2+1; k++) {
        out_data_buf[k] = out_data_ptr1[k*out_stride];
      }

      NaiveDft(reference_fft, make_cspan(in_data_buf), nfft);

      LOG_LINE << "Reference data: ";
      for (int k = 0; k < nfft/2+1; k++) {
        LOG_LINE << " (" << reference_fft[k].real() << ", " << reference_fft[k].imag() << "),";
      }
      LOG_LINE << std::endl;

      LOG_LINE << "Actual data: ";
      for (int k = 0; k < nfft/2+1; k++) {
        LOG_LINE << " (" << out_data_buf[k].real() << ", " << out_data_buf[k].imag() << "),";
      }
      LOG_LINE << std::endl;

      CompareFfts(make_cspan(reference_fft), make_cspan(out_data_buf));
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
