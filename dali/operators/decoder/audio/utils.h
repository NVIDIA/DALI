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

#ifndef DALI_OPERATORS_DECODER_AUDIO_UTILS_H_
#define DALI_OPERATORS_DECODER_AUDIO_UTILS_H_

#include <functional>
#include <utility>
#include <vector>
#include <cassert>
#include <cmath>
#include "dali/core/static_switch.h"
#include "dali/core/convert.h"
#include "dali/core/error_handling.h"
#include "dali/core/span.h"

namespace dali {

namespace resampling {
double Hann(double x) {
  return 0.5 * (1 + std::cos(x * M_PI_2));
}

struct sinc_coeffs {
  void init(int coeffs, int lobes = 3, std::function<double(double)> envelope = Hann) {
    float scale = 2.0f * lobes / (coeffs - 1);
    float scale_envelope = 2.0f / coeffs;
    this->coeffs = coeffs;
    this->lobes = lobes;
    lookup.resize(coeffs + 2);  // add zeros
    center = (coeffs - 1) * 0.5f;
    for (int i = 0; i < coeffs; i++) {
      float x = (i - center) * scale;
      float y = (i - center) * scale_envelope;
      float w = sinc(x) * envelope(y);
      lookup[i + 1] = w;
      std::cerr << i << ": " << w << "\n";
    }
    center++;  // allow for leading zero
    this->scale = 1 / scale;
  }


  std::pair<int, int> input_range(float x) const {
    int i0 = std::ceil(x) - lobes;
    int i1 = std::floor(x) + lobes;
    return {i0, i1};
  }


  float operator()(float x) const {
    float fi = x * scale + center;
    int i = std::floor(fi);
    float di = fi - i;
    assert(i >= 0 && i < (int) lookup.size());  // NOLINT
    return lookup[i] + di * (lookup[i + 1] - lookup[i]);
  }


  float scale = 1, center = 1;
  int lobes = 0, coeffs = 0;
  std::vector<float> lookup;
};


/**
 * Resampling for audio buffer
 */
void resample_sinc(
        float *out, int64_t n_out, double out_rate,
        const float *in, int64_t n_in, double in_rate,
        int lobes = 3) {
  sinc_coeffs coeffs;
  coeffs.init(1024, lobes);
  int64_t in_pos = 0;
  int64_t block = 1 << 10;  // still leaves 13 significant bits for fractional part
  double scale = in_rate / out_rate;
  float fscale = scale;
  for (int64_t out_block = 0; out_block < n_out; out_block += block) {
    int64_t block_end = std::min(out_block + block, n_out);
    double in_block_f = (out_block + 0.5) * scale - 0.5;
    int64_t in_block_i = std::floor(in_block_f);
    float in_pos = in_block_f - in_block_i;
    const float *in_block_ptr = in + in_block_i;
    for (int64_t out_pos = out_block; out_pos < block_end; out_pos++, in_pos += fscale) {
      int i0 = std::ceil(in_pos) - lobes;
      int i1 = std::floor(in_pos) + lobes;
      if (i0 + in_block_i < 0)
        i0 = -in_block_i;
      if (i1 + in_block_i >= n_in)
        i1 = n_in - 1 - in_block_i;
      float f = 0;
      float x = i0 - in_pos;
      for (int i = i0; i <= i1; i++, x++) {
        assert(in_block_ptr + i >= in && in_block_ptr + i < in + n_in);
        float w = coeffs(x);
        f += in_block_ptr[i] * w;
      }
      assert(out_pos >= 0 && out_pos < n_out);
      out[out_pos] = f;
    }
  }
}
}  // namespace resampling


}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_AUDIO_UTILS_H_
