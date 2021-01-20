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

#ifndef DALI_KERNELS_AUDIO_MEL_SCALE_MEL_SCALE_H_
#define DALI_KERNELS_AUDIO_MEL_SCALE_MEL_SCALE_H_

#include <vector>
#include <cmath>
#include "dali/core/force_inline.h"
#include "dali/kernels/audio/mel_scale/mel_filter_bank_args.h"

namespace dali {
namespace kernels {
namespace audio {

template <typename T>
struct HtkMelScale {
  DALI_FORCEINLINE T hz_to_mel(T hz) {
    // equivalent to `2595.0 * std::log10(1 + hz / 700.0)`
    return T(1127) * std::log(T(1) + hz / T(700));
  }

  DALI_FORCEINLINE T mel_to_hz(T mel) {
    // equivalent to `700.0 * (std::pow(10, mel / 2595.0) - 1.0)`
    return T(700) * (std::exp(mel / T(1127)) - T(1));
  }
};

template <typename T>
struct SlaneyMelScale {
  static constexpr T freq_low = 0;
  static constexpr T fsp = 200.0 / 3.0;

  static constexpr T min_log_hz = 1000.0;
  static constexpr T min_log_mel = (min_log_hz - freq_low) / fsp;
  static constexpr T step_log = 0.068751777;  // Equivalent to std::log(6.4) / 27.0;

  DALI_FORCEINLINE T hz_to_mel(T hz) {
    T mel = 0;
    if (hz >= min_log_hz) {
      // non-linear scale
      mel = min_log_mel + std::log(hz / min_log_hz) / step_log;
    } else {
      // linear scale
      mel = (hz - freq_low) / fsp;
    }

    return mel;
  }

  DALI_FORCEINLINE T mel_to_hz(T mel) {
    T hz = 0;
    if (mel >= min_log_mel) {
      // non linear scale
      hz = min_log_hz * std::exp(step_log * (mel - min_log_mel));
    } else {
      // linear scale
      hz = freq_low + mel * fsp;
    }
    return hz;
  }
};

template <typename T>
class MelFilterImplBase {
 public:
  template <typename MelScale>
  MelFilterImplBase(MelScale mel_scale, const MelFilterBankArgs &args): args_(args) {
    assert(args.sample_rate > 0);

    assert(args.freq_low >= 0 && args.freq_low <= args.sample_rate / 2);
    mel_low_ = mel_scale.hz_to_mel(args.freq_low);

    assert(args.freq_high >= 0 && args.freq_high <= args.sample_rate / 2);
    mel_high_ = mel_scale.hz_to_mel(args.freq_high);

    int nfilter = args.nfilter;
    assert(nfilter > 0);

    int nfft = args.nfft;
    assert(nfft > 0);

    hz_step_ = static_cast<double>(args.sample_rate) / nfft;
    mel_delta_ = (mel_high_ - mel_low_) / (nfilter + 1);

    fftbin_size_ = nfft / 2 + 1;
    double inv_hz_step = 1.0 / hz_step_;
    fftbin_start_ = std::ceil(args.freq_low * inv_hz_step);
    assert(fftbin_start_ >= 0);
    fftbin_end_ = std::floor(args.freq_high * inv_hz_step);
    if (fftbin_end_ > fftbin_size_ - 1)
      fftbin_end_ = fftbin_size_ - 1;

    weights_down_.resize(fftbin_size_);
    norm_factors_.resize(nfilter, T(1));
    double mel0 = mel_low_, mel1 = mel_low_ + mel_delta_;

    int fftbin = fftbin_start_;
    double f = fftbin * hz_step_;

    int last_interval = nfilter;
    for (int interval = 0; interval <= last_interval;
         interval++, mel0 = mel1, mel1 += mel_delta_) {
      if (interval == last_interval)
        mel1 = mel_high_;
      double f0 = mel_scale.mel_to_hz(mel0),
             f1 = mel_scale.mel_to_hz(mel1);
      if (args.normalize && interval < nfilter) {
        // Filters are normalized so that they have constant energy per band
        double f2 = mel_scale.mel_to_hz(mel1 + mel_delta_);
        norm_factors_[interval] = 2.0 / (f2 - f0);
      }

      double slope = 1. / (f1 - f0);
      for (; fftbin <= fftbin_end_ && f < f1; fftbin++, f = fftbin * hz_step_) {
        weights_down_[fftbin] = (f1 - f) * slope;
      }
    }
  }

  const MelFilterBankArgs& Args() const {
    return args_;
  }

 protected:
  MelFilterBankArgs args_;
  std::vector<T> weights_down_;
  std::vector<T> norm_factors_;
  int fftbin_start_ = -1, fftbin_end_ = -1;
  int fftbin_size_;
  double mel_low_, mel_high_;
  double hz_step_, mel_delta_;
};

#define USE_MEL_FILTER_IMPL_MEMBERS(T) \
  using MelFilterImplBase<T>::args_; \
  using MelFilterImplBase<T>::fftbin_start_; \
  using MelFilterImplBase<T>::fftbin_end_; \
  using MelFilterImplBase<T>::mel_low_; \
  using MelFilterImplBase<T>::mel_high_; \
  using MelFilterImplBase<T>::fftbin_size_; \
  using MelFilterImplBase<T>::weights_down_; \
  using MelFilterImplBase<T>::norm_factors_; \
  using MelFilterImplBase<T>::mel_delta_; \
  using MelFilterImplBase<T>::hz_step_

}  // namespace audio
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_AUDIO_MEL_SCALE_MEL_SCALE_H_
