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

#ifndef DALI_TEST_TEST_SOUND_GENERATOR_H_
#define DALI_TEST_TEST_SOUND_GENERATOR_H_

#include <math.h>
#include <random>

namespace dali {
namespace testing {

/**
 * @brief Adds an enveloped sinewave to data in `out`.
 *
 * @param inout the sound to add the wave to
 * @param length length, in samples, of the generates segment
 * @param freq frequency relative to Nyquist
 * @param amplitude self-explanatory
 */
inline void GenerateTestSound(float *inout, int length, float freq, float amplitude) {
  float m = M_PI * freq;
  int fade = length/4;
  float ampl[8];

  for (int h = 0; h < 8; h++)
    ampl[h] = pow(0.5, h);

  auto signal = [&](int i) {
    float v = 0;
    for (int h = 0; h < 8; h++) {
      float phase = i * m * (2*h+1);  // generate odd harmonics
      v += sin(phase) * ampl[h];
    }
    return v * amplitude;
  };

  int i = 0;
  for (; i < fade; i++) {
    float envelope = (1 - cos(M_PI*i/fade)) * 0.5f;
    inout[i] += signal(i) * envelope;
  }

  for (; i < length - fade; i++) {
    inout[i] += signal(i);
  }

  for (; i < length; i++) {
    float envelope = (1 - cos(M_PI*(length - i)/fade)) * 0.5f;
    inout[i] += signal(i) * envelope;
  }
}

/**
 * @brief Generates a test recording
 *
 * The sound consists of `num_sounds` sounds (@see GenerateTestSound) + Gaussian noise.
 */
template <typename RNG>
void GenerateTestWave(RNG &rng, float *out, int length, int num_sounds, int max_sound_length,
                      float noise_level = 0.01f) {
  std::normal_distribution<float> noise(0, noise_level);
  std::uniform_int_distribution<int> lengths(max_sound_length/10, max_sound_length);
  std::uniform_real_distribution<float> freqs(1e-3f, 0.3f);
  std::uniform_real_distribution<float> ampls(0.1f, 1.0f);
  for (int i = 0; i < length; i++)
    out[i] = noise(rng);
  for (int i = 0; i < num_sounds; i++) {
    int l = lengths(rng);
    int pos = std::uniform_int_distribution<int>(0, length - l)(rng);
    GenerateTestSound(out + pos, l, freqs(rng), ampls(rng));
  }
}

}  // namespace testing
}  // namespace dali

#endif  // DALI_TEST_TEST_SOUND_GENERATOR_H_
