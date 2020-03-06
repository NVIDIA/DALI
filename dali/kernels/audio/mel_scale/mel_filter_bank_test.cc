// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/kernels/audio/mel_scale/mel_filter_bank_test.h"


namespace dali {
namespace kernels {
namespace audio {
namespace test {

std::vector<std::vector<float>> ReferenceFilterBanks(int nfilter, int nfft, float sample_rate,
                                                     float low_freq, float high_freq) {
  HtkMelScale<float> mel_scale;

  std::vector<std::vector<float>> fbanks(nfilter);
  auto low_mel = mel_scale.hz_to_mel(low_freq);
  auto high_mel = mel_scale.hz_to_mel(high_freq);

  float delta_mel = (high_mel - low_mel) / (nfilter + 1);
  assert(nfilter > 0);
  std::vector<float> mel_points(nfilter+2, 0.0f);
  mel_points[0] = mel_scale.hz_to_mel(low_freq);
  for (int i = 1; i < nfilter+1; i++) {
    mel_points[i] = mel_points[i-1] + delta_mel;
  }
  mel_points[nfilter+1] = mel_scale.hz_to_mel(high_freq);

  std::vector<float> fftfreqs(nfft/2+1, 0.0f);
  for (int i = 0; i < nfft/2+1; i++) {
    fftfreqs[i] = i * sample_rate / nfft;
  }

  std::vector<float> freq_grid(mel_points.size(), 0.0f);
  freq_grid[0] = low_freq;
  for (int i = 1; i < nfilter+1; i++) {
    freq_grid[i] = mel_scale.mel_to_hz(mel_points[i]);
  }
  freq_grid[nfilter+1] = high_freq;

  for (int j = 0; j < nfilter; j++) {
    auto &fbank = fbanks[j];
    fbank.resize(nfft/2+1, 0.0f);
    for (int i = 0; i < nfft/2+1; i++) {
      auto f = fftfreqs[i];
      if (f < low_freq || f > high_freq) {
        fbank[i] = 0.0f;
      } else {
        auto upper = (f - freq_grid[j]) / (freq_grid[j+1] - freq_grid[j]);
        auto lower = (freq_grid[j+2] - f) / (freq_grid[j+2] - freq_grid[j+1]);
        fbank[i] = std::max(0.0f, std::min(upper, lower));
      }
    }
  }

  for (int j = 0; j < nfilter; j++) {
    LOG_LINE << "Filter " << j << " :";
    auto &fbank = fbanks[j];
    for (float f : fbank) {
      LOG_LINE << " " << f;
    }
    LOG_LINE << std::endl;
  }
  return fbanks;
}

}  // namespace test
}  // namespace audio
}  // namespace kernels
}  // namespace dali
