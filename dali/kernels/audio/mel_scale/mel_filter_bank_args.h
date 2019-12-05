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

#ifndef DALI_KERNELS_AUDIO_MEL_SCALE_MEL_FILTER_BANK_ARGS_H_
#define DALI_KERNELS_AUDIO_MEL_SCALE_MEL_FILTER_BANK_ARGS_H_

namespace dali {
namespace kernels {
namespace audio {

enum class MelScaleFormula {
  Slaney,
  HTK
};

struct MelFilterBankArgs {
  /// @brief Signal's sampling rate
  float sample_rate = 44100.0f;

  /// @brief Minimum frequency
  float freq_low = 0.0f;

  /// @brief Maximum frequency (default is Nyquist frequency)
  float freq_high = sample_rate / 2;

  /// @brief Number of mel filter banks
  int nfilter = 128;

  /// @brief Axis corresponding to the frequency domain in the input spectrogram
  int axis = -1;

  /// @brief Number of bins used in the FFT to build the input spectrogram
  //         It should match the shape of the input data: `input.shape[axis] == nfft/2 + 1`
  int nfft = -1;

  /// @brief Determines the formula used in for mel_to_hz and hz_to_mel calculations
  MelScaleFormula mel_formula = MelScaleFormula::Slaney;

  /// @brief Determines whether to normalize the filter weights by the width of the mel band
  bool normalize = true;

  bool operator==(const MelFilterBankArgs &oth) const {
    return sample_rate == oth.sample_rate
        && freq_low  == oth.freq_low
        && freq_high  == oth.freq_high
        && nfilter == oth.nfilter
        && axis == oth.axis
        && nfft == oth.nfft
        && mel_formula == oth.mel_formula
        && normalize == oth.normalize;
  }

  bool operator!=(const MelFilterBankArgs &oth) const {
    return !operator==(oth);
  }
};

}  // namespace audio
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_AUDIO_MEL_SCALE_MEL_FILTER_BANK_ARGS_H_
