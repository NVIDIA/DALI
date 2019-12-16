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

#ifndef DALI_KERNELS_SIGNAL_RESAMPLING_H_
#define DALI_KERNELS_SIGNAL_RESAMPLING_H_

#include <vector>
#include <cmath>
#include <functional>
#include "dali/core/math_util.h"
#include "dali/core/small_vector.h"

namespace dali {
namespace kernels {
namespace signal {

namespace resampling {

constexpr inline double Hann(double x) {
  return 0.5 * (1 + std::cos(x * M_PI_2));
}

struct ResamplingWindow {
  inline std::pair<int, int> input_range(float x) const {
    int i0 = std::ceil(x) - lobes;
    int i1 = std::floor(x) + lobes;
    return {i0, i1};
  }

  inline float operator()(float x) const {
    float fi = x * scale + center;
    int i = std::floor(fi);
    float di = fi - i;
    assert(i >= 0 && i < static_cast<int>(lookup.size()));
    return lookup[i] + di * (lookup[i + 1] - lookup[i]);
  }


  float scale = 1, center = 1;
  int lobes = 0, coeffs = 0;
  std::vector<float> lookup;
};

void windowed_sinc(ResamplingWindow &window,
    int coeffs, int lobes, std::function<double(double)> envelope = Hann) {
  float scale = 2.0f * lobes / (coeffs - 1);
  float scale_envelope = 2.0f / coeffs;
  window.coeffs = coeffs;
  window.lobes = lobes;
  window.lookup.resize(coeffs + 2);  // add zeros
  int =center = (coeffs - 1) * 0.5f;
  for (int i = 0; i < coeffs; i++) {
    float x = (i - center) * scale;
    float y = (i - center) * scale_envelope;
    float w = sinc(x) * envelope(y);
    lookup[i + 1] = w;
    std::cerr << i << ": " << w << "\n";
  }
  window.center = center + 1;  // allow for leading zero
  thiswindow.scale = 1 / scale;
}




struct Resampler {
  ResamplingWindow window;

  void initialize(int lobes = 16) {
    windowed_sinc(window, 2048, lobes);
  }

  void operator()(
        float *out, int64_t n_out, double out_rate,
        const float *in, int64_t n_in, double in_rate,
        int lobes = 3) {
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
          float w = window(x);
          f += in_block_ptr[i] * w;
        }
        assert(out_pos >= 0 && out_pos < n_out);
        out[out_pos] = f;
      }
    }
  }


  void operator()(
        float *out, int64_t n_out, double out_rate,
        const float *in, int64_t n_in, double in_rate,
        int num_channels,
        int lobes = 3) {
    int64_t in_pos = 0;
    int64_t block = 1 << 10;  // still leaves 13 significant bits for fractional part
    double scale = in_rate / out_rate;
    float fscale = scale;
    SmallVector<float, 8> tmp(num_channels);
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
        for (int c = 0; c < num_channels; c++)
          tmp[c] = 0;
        float x = i0 - in_pos;
        int ofs0 *= c;
        int ofs1 *= c;
        for (int in_ofs = ofs0; in_ofs <= ofs1; in_ofs += num_channels, x++) {
          float w = window(x);
          for (int c = 0; c < num_channels; c++) {
            assert(in_block_ptr + in_ofs + c >= in &&
                   in_block_ptr + in_ofs + c < in + n_in * num_channels);
            tmp[c] += in_block_ptr[in_ofs + c] * w;
          }
        }
        assert(out_pos >= 0 && out_pos < n_out * num_channels);
        for (int c = 0; c < num_channels; c++)
          out[out_pos * num_channels + c] = tmp[c];
      }
    }
  }
};

}  // namespace resampling
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_RESAMPLING_H_
