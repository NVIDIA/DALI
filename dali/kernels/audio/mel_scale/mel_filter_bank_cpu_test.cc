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
#include "dali/kernels/audio/mel_scale/mel_filter_bank_cpu_impl.h"
#include "dali/kernels/audio/mel_scale/mel_filter_bank_cpu.h"
#include "dali/kernels/common/utils.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"

#undef LOG_LINE
#define LOG_LINE std::cout

namespace dali {
namespace kernels {
namespace audio {
namespace test {

class MelScaleCpuTest : public::testing::TestWithParam<
  std::tuple<std::array<int64_t, 2> /* data_shape */>> {
 public:
  MelScaleCpuTest()
    : data_shape_(std::get<0>(GetParam()))
    , data_(volume(data_shape_))
    , in_view_(data_.data(), data_shape_) {}

  ~MelScaleCpuTest() override = default;

 protected:
  void SetUp() final {
    std::mt19937 rng;
    UniformRandomFill(in_view_, rng, 0.0, 1.0);
  }
  TensorShape<2> data_shape_;
  std::vector<float> data_;
  OutTensorCPU<float, 2> in_view_;
};

template <typename T>
void print_data(const OutTensorCPU<T, 2>& data_view) {
  auto sh = data_view.shape;
  for (int i0 = 0; i0 < sh[0]; i0++) {
    for (int i1 = 0; i1 < sh[1]; i1++) {
      int k = i0 * sh[1] + i1;
      LOG_LINE << " " << data_view.data[k];
    }
    LOG_LINE << "\n";
  }
}

float hz_to_mel(float hz) {
  return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float mel_to_hz(float mel) {
  return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

std::vector<std::vector<float>> ReferenceFilterBanks(int nfilter, int nfft, float sample_rate,
                                                     float low_freq, float high_freq) {
  std::vector<std::vector<float>> fbanks(nfilter);
  auto low_mel = hz_to_mel(low_freq);
  auto high_mel = hz_to_mel(high_freq);

  float delta_mel = (high_mel - low_mel) / (nfilter + 1);
  std::vector<float> mel_points(nfilter+2, 0.0f);
  for (int i = 0; i < static_cast<int>(mel_points.size()); i++) {
    mel_points[i] = i * delta_mel;
  }

  std::vector<float> fftfreqs(nfft/2+1, 0.0f);
  for (int i = 0; i < nfft/2+1; i++) {
    fftfreqs[i] = i * sample_rate / nfft;
  }

  std::vector<float> freq_grid(mel_points.size(), 0.0f);
  for (int i = 0; i < static_cast<int>(mel_points.size()); i++) {
    freq_grid[i] = mel_to_hz(mel_points[i]);
  }

  for (int j = 0; j < nfilter; j++) {
    auto &fbank = fbanks[j];
    fbank.resize(nfft/2+1, 0.0f);
    for (int i = 0; i < nfft/2+1; i++) {
      auto f = fftfreqs[i];
      auto upper = (f - freq_grid[j]) / (freq_grid[j+1] - freq_grid[j]);
      auto lower = (freq_grid[j+2] - f) / (freq_grid[j+2] - freq_grid[j+1]);
      fbank[i] = std::max(0.0f, std::min(upper, lower));
    }

    /*for (i = std::floor(bins[j]); i < std::ceil(bins[j+1]); i++) {
      std::cout << "A Visit " << i << "\n";
      auto ii = i + 0.5f;
      fbank[i] = std::max(0.0f, (ii - bins[j]) / (bins[j+1] - bins[j]));
    }

    for (i = std::floor(bins[j+1]); i < std::ceil(bins[j+2]); i++) {
      std::cout << "B Visit " << i << "\n";
      auto ii = i + 0.5f;
      fbank[i] = std::max(0.0f, (bins[j+2] - ii) / (bins[j+2] - bins[j+1]));
    }*/
  }

  for (int j = 0; j < nfilter; j++) {
    LOG_LINE << "Filter " << j << " :";
    auto &fbank = fbanks[j];
    for (int i = 0; i < static_cast<int>(fbank.size()); i++) {
      LOG_LINE << " " << fbank[i];
    }
    LOG_LINE << std::endl;
  }
  return fbanks;
}

TEST_P(MelScaleCpuTest, MelScaleCpuTest) {
  using T = float;
  constexpr int Dims = 2;

  auto shape = in_view_.shape;

  int axis = 0;
  int nfilters = 4;
  int nfft = (shape[axis]-1)*2;
  int nwin = shape[axis+1];

  auto out_shape = in_view_.shape;
  out_shape[axis] = nfilters;
  auto out_size = volume(out_shape);

  std::vector<T> expected_out(out_size);
  auto expected_out_view = OutTensorCPU<T, Dims>(expected_out.data(), out_shape.to_static<Dims>());

  T sample_rate = 16000;
  T mel_low = hz_to_mel(0.0f);
  T mel_high = hz_to_mel(sample_rate / 2);

  T mel_delta = (mel_high - mel_low) / (nfilters + 1);
  T mel = mel_low;
  for (int i = 0; i < nfilters+1; i++, mel += mel_delta) {
    LOG_LINE << "Mel freq " << i << " : " << mel << " \n";
  }
  LOG_LINE << "Mel freq " << nfilters+1 << " : " << mel_high << " \n";

  LOG_LINE << "FFT bin to Mels:";
  for (int k = 0; k < nfft / 2 + 1; k++) {
    LOG_LINE << "\n k " << k << " : " << (k * sample_rate / nfft);
  }
  LOG_LINE << "\n";

  auto fbanks = ReferenceFilterBanks(nfilters, nfft, sample_rate, 0.0f, sample_rate/2.0f);
  std::vector<T> expected_out2(out_size, 0.0f);
  auto expected_out_view2 = OutTensorCPU<T, Dims>(expected_out2.data(), out_shape.to_static<Dims>());
  for (int j = 0; j < nfilters; j++) {
    for (int t = 0; t < nwin; t++) {
      auto &out_val = expected_out_view2.data[j*nwin+t];
      for (int i = 0; i < nfft/2+1; i++) {
        out_val += fbanks[j][i] * in_view_.data[i*nwin+t];
      }
    }
  }

  // In the outer loop we travel at a linearly spaced frequency grid in the mel scale
  // Each triangular filter is defined by three points in this grid (left, center, right)
  // For each iteration we process a range between two mel frequencies in the grid, calculating
  // the contribution of each FFT bin to 2 triangular filter (one is in the negative slope region
  // and the other in the positive slope region), except for the first and last iteration.
  // In total, we do a single pass on every FFT bin column
  //
  // For every FFT bin we compute the weight for each filter and travel through the row, computing
  // the contributions on every window of the spectrogram (horizontal axis)
  //

  T mel0 = mel_low;
  T mel1 = mel_low + mel_delta;
  T *out_ = &expected_out_view.data[0];

  // Index of fft bins [0, nfft/2+1)
  int k = 0;
  // Frequency increment in Hz between FFT bins
  auto hz_step = sample_rate / nfft;
  // Frequency start in Hz
  auto hz = 0.5 * hz_step;  // centered bins
/*
  int last_interval = nfilters;
  for (int interval = 0, filter_up = 0, filter_down = -1;
       interval <= last_interval;
       interval++, mel0 = mel1, mel1 += mel_delta, filter_up++, filter_down++) {
    T slope = 1.0 / (mel1 - mel0);
    if (interval == last_interval)
      mel1 = mel_high;
    LOG_LINE << "Enter interval " << interval << " from mel " << mel0 << " up to mel " << mel1 << "\n";

    for (; k < nfft/2+1; k++, hz += hz_step) {
      auto mel = hz_to_mel(hz);
      LOG_LINE << "k " << k << " current mel " << mel << " ";
      if (mel > mel1) {
        LOG_LINE << "... Moving on to next filter\n";
        break;
      }
      auto *in_row_start = &in_view_.data[k*nwin];
      T weight_up = 0.0, weight_down = 0.0;
      if (filter_down >= 0) {
        LOG_LINE << "... filter down! ";
        weight_down = (mel1-mel) * slope;
        auto *out_row_start = &expected_out_view.data[filter_down*nwin];
        LOG_LINE << "(" << mel << " - " << mel0 << ") / " << (mel1 - mel0)
                  << " = " << weight_down << std::endl;
        for (int t = 0; t < nwin; t++)
          out_row_start[t] += weight_down * in_row_start[t];
      }

      if (filter_up < nfilters) {
        LOG_LINE << "... filter up! ";
        weight_up = (mel-mel0) * slope;
        auto *out_row_start = &expected_out_view.data[filter_up*nwin];
        LOG_LINE << "(" << mel << " - " << mel0 << ") / " << (mel1 - mel0)
                  << " = " << weight_up << std::endl;
        for (int t = 0; t < nwin; t++)
          out_row_start[t] += weight_up * in_row_start[t];
      }

      LOG_LINE << "Sum is " << weight_down + weight_up << "\n";
    }
  }*/

  MelFilterBankImpl<T> fbank(nfilters, nfft, sample_rate);
  fbank.Compute(expected_out_view.data, in_view_.data, nwin);

  LOG_LINE << "in:\n";
  print_data(in_view_);

  LOG_LINE << "expected out:\n";
  print_data(expected_out_view);

  LOG_LINE << "expected out 2:\n";
  print_data(expected_out_view2);
}

INSTANTIATE_TEST_SUITE_P(MelScaleCpuTest, MelScaleCpuTest, testing::Combine(
    testing::Values(std::array<int64_t, 2>{33, 10})));  // shape

}  // namespace test
}  // namespace audio
}  // namespace kernels
}  // namespace dali
