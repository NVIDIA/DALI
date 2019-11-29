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

#ifndef DALI_KERNELS_AUDIO_MEL_SCALE_MEL_FILTER_BANK_IMPL_H_
#define DALI_KERNELS_AUDIO_MEL_SCALE_MEL_FILTER_BANK_IMPL_H_

#include <cmath>
#include <vector>

namespace dali {
namespace kernels {
namespace audio {

template <typename T>
T hz_to_mel(T hz) {}

template <>
float hz_to_mel(float hz) {
  // equivalent to `2595.0 * std::log10(1 + hz / 700.0)`
  return 1127.0f * std::log(1.0f + hz / 700.0f);
}

template <>
double hz_to_mel(double hz) {
  // equivalent to `2595.0 * std::log10(1 + hz / 700.0)`
  return 1127.0 * std::log(1.0 + hz / 700.0);
}

template <typename T>
T mel_to_hz(T mel);

template <>
float mel_to_hz(float mel) {
  // equivalent to `700.0 * (std::pow(10, mel / 2595.0) - 1.0)`
  return 700.0f * (std::exp(mel / 1127.0f) - 1.0f);
}

template <>
double mel_to_hz(double mel) {
  // equivalent to `700.0 * (std::pow(10, mel / 2595.0) - 1.0)`
  return 700.0 * (std::exp(mel / 1127.0) - 1.0);
}


template <typename T>
class MelFilterBankImpl {
 public:
  MelFilterBankImpl(int nfilter, int nfft, T sample_rate, T freq_low = 0, T freq_high = 0)
      : nfilter_(nfilter), nfft_(nfft), sample_rate_(sample_rate)
      , freq_low_(freq_low), freq_high_(freq_high > 0 ? freq_high : sample_rate / 2) {
    mel_low_ = hz_to_mel(freq_low_);
    mel_high_ = hz_to_mel(freq_high_);
    hz_step_ = sample_rate_ / nfft_;
    mel_delta_ = (mel_high_ - mel_low_) / (nfilter_ + 1);
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
  void Compute(T* out, const T* in, int64_t nwindows,
               int64_t out_stride = -1, int64_t in_stride = -1) {
    if (out_stride <= 0)
      out_stride = nwindows;

    if (in_stride <= 0)
      in_stride = nwindows;

    int last_interval = nfilter_;
    T mel0 = mel_low_;
    T mel1 = mel_low_ + mel_delta_;
    int fftbin = 0;
    T hz = 0.5 * hz_step_;  // centered
    for (int interval = 0, filter_up = 0, filter_down = -1;
         interval <= last_interval;
         interval++, mel0 = mel1, mel1 += mel_delta_, filter_up++, filter_down++) {
      T slope = 1.0 / (mel1 - mel0);
      if (interval == last_interval)
        mel1 = mel_high_;

      for (; fftbin < nfft_/2+1; fftbin++, hz += hz_step_) {
        auto mel = hz_to_mel(hz);
        if (mel > mel1)
          break;

        auto *in_row_start = in + fftbin * in_stride;
        T weight_up = 0.0, weight_down = 0.0;
        if (filter_down >= 0) {
          weight_down = (mel1 - mel) * slope;
          auto *out_row_start = out + filter_down * out_stride;
          for (int t = 0; t < nwindows; t++)
            out_row_start[t] += weight_down * in_row_start[t];
        }

        if (filter_up < nfilter_) {
          weight_up = (mel - mel0) * slope;
          auto *out_row_start = out + filter_up * out_stride;
          for (int t = 0; t < nwindows; t++)
            out_row_start[t] += weight_up * in_row_start[t];
        }
      }
    }
  }

 private:
  int nfilter_;
  int nfft_;
  T sample_rate_;
  T freq_low_;
  T freq_high_;
  T mel_low_;
  T mel_high_;
  T hz_step_;
  T mel_delta_;
};

}  // namespace audio
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_AUDIO_MEL_SCALE_MEL_FILTER_BANK_IMPL_H_