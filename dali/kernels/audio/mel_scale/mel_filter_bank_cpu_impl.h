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
class MelFilterBankCpuImpl {
 public:
  MelFilterBankCpuImpl(int nfilter, int nfft, T sample_rate, T freq_low = 0, T freq_high = 0)
      : nfilter_(nfilter), nfft_(nfft), sample_rate_(sample_rate)
      , freq_low_(freq_low), freq_high_(freq_high > 0 ? freq_high : sample_rate / 2) {
    mel_low_ = hz_to_mel(freq_low_);
    mel_high_ = hz_to_mel(freq_high_);
    hz_step_ = sample_rate_ / nfft_;
    mel_delta_ = (mel_high_ - mel_low_) / (nfilter_ + 1);

    int64_t fftbin = 0;
    int64_t fftbin_size = nfft_ / 2 + 1;
    weights_down_.resize(fftbin_size);
    intervals_.resize(fftbin_size);
    T mel0 = mel_low_, mel1 = mel_low_ + mel_delta_;
    T f = centered_ ? T(0.5) * hz_step_ : T(0);
    int last_interval = nfilter_;
    for (int64_t interval = 0; interval <= last_interval;
         interval++, mel0 = mel1, mel1 += mel_delta_) {
      if (interval == last_interval)
        mel1 = mel_high_;
      T f0 = mel_to_hz(mel0), f1 = mel_to_hz(mel1);
      T slope = T(1) / (f1 - f0);
      for (; fftbin < fftbin_size && f < f1; fftbin++, f += hz_step_) {
        weights_down_[fftbin] = (f1 - f) * slope;;
        intervals_[fftbin] = interval;
      }
    }
  }

  // In the outer loop we travel at a linearly spaced frequency grid in the mel scale
  // Each triangular filter is defined by three points in this grid (left, center, right)
  // For each iteration we process a range between two mel frequencies in the grid, calculating
  // the contribution of each FFT bin to 2 triangular filters (one is in the negative slope region
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

    std::memset(out, 0, sizeof(T) * nfilter_ * nwindows);
    int fftbin_size = nfft_ / 2 + 1;
    for (int64_t fftbin = 0; fftbin < fftbin_size; fftbin++) {
      auto *in_row_start = in + fftbin * in_stride;
      auto filter_up = intervals_[fftbin];
      auto weight_up = T(1) - weights_down_[fftbin];
      auto filter_down = filter_up - 1;
      auto weight_down = weights_down_[fftbin];

      if (filter_down >= 0) {
        auto *out_row_start = out + filter_down * out_stride;
        for (int t = 0; t < nwindows; t++) {
          out_row_start[t] += weight_down * in_row_start[t];
        }
      }

      if (filter_up < nfilter_) {
        auto *out_row_start = out + filter_up * out_stride;
        for (int t = 0; t < nwindows; t++) {
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
  std::vector<T> weights_down_;
  std::vector<int> intervals_;
  bool centered_ = false;
};

}  // namespace audio
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_AUDIO_MEL_SCALE_MEL_FILTER_BANK_IMPL_H_