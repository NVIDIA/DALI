// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifdef __SSE2__
#include <emmintrin.h>
#endif
#include <cmath>
#include <functional>
#include <utility>
#include <vector>
#include "dali/core/math_util.h"
#include "dali/core/small_vector.h"
#include "dali/core/convert.h"
#include "dali/core/static_switch.h"

namespace dali {
namespace kernels {
namespace signal {

namespace resampling {

inline double Hann(double x) {
  return 0.5 * (1 + std::cos(x * M_PI));
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

#ifdef __SSE2__
  inline __m128 operator()(__m128 x) const {
    __m128 fi = _mm_add_ps(x * _mm_set1_ps(scale), _mm_set1_ps(center));
    __m128i i = _mm_cvtps_epi32(fi);
    __m128 fifloor = _mm_cvtepi32_ps(i);
    __m128 di = _mm_sub_ps(fi, fifloor);
    int idx[4];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(idx), i);
    __m128 curr = _mm_setr_ps(lookup[idx[0]],   lookup[idx[1]],
                              lookup[idx[2]],   lookup[idx[3]]);
    __m128 next = _mm_setr_ps(lookup[idx[0]+1], lookup[idx[1]+1],
                              lookup[idx[2]+1], lookup[idx[3]+1]);
    return _mm_add_ps(curr, _mm_mul_ps(di, _mm_sub_ps(next, curr)));
  }
#endif


  float scale = 1, center = 1;
  int lobes = 0, coeffs = 0;
  std::vector<float> lookup;
};

inline void windowed_sinc(ResamplingWindow &window,
    int coeffs, int lobes, std::function<double(double)> envelope = Hann) {
  assert(coeffs > 1 && lobes > 0 && "Degenerate parameters specified.");
  float scale = 2.0f * lobes / (coeffs - 1);
  float scale_envelope = 2.0f / coeffs;
  window.coeffs = coeffs;
  window.lobes = lobes;
  window.lookup.resize(coeffs + 2);  // add zeros
  int center = (coeffs - 1) * 0.5f;
  for (int i = 0; i < coeffs; i++) {
    float x = (i - center) * scale;
    float y = (i - center) * scale_envelope;
    float w = sinc(x) * envelope(y);
    window.lookup[i + 1] = w;
  }
  window.center = center + 1;  // allow for leading zero
  window.scale = 1 / scale;
}


inline int64_t resampled_length(int64_t in_length, double in_rate, double out_rate) {
  return std::ceil(in_length * out_rate / in_rate);
}

struct Resampler {
  ResamplingWindow window;

  void Initialize(int lobes = 16, int lookup_size = 2048) {
    windowed_sinc(window, lookup_size, lobes);
  }

  /**
   * @brief Resample single-channel signal and convert to Out
   *
   * Calculates a range of resampled signal.
   * The function can seamlessly resample the input and produce the result in chunks.
   * To reuse memory and still simulate chunk processing, adjust the in/out pointers.
   */
  template <typename Out>
  void Resample(
        Out *__restrict__ out, int64_t out_begin, int64_t out_end, double out_rate,
        const float *__restrict__ in, int64_t n_in, double in_rate) const {
    assert(out_rate > 0 && in_rate > 0 && "Sampling rate must be positive");
    int64_t in_pos = 0;
    int64_t block = 1 << 10;  // still leaves 13 significant bits for fractional part
    double scale = in_rate / out_rate;
    float fscale = scale;
    for (int64_t out_block = out_begin; out_block < out_end; out_block += block) {
      int64_t block_end = std::min(out_block + block, out_end);
      double in_block_f = out_block * scale;
      int64_t in_block_i = std::floor(in_block_f);
      float in_pos = in_block_f - in_block_i;
      const float *__restrict__ in_block_ptr = in + in_block_i;
      for (int64_t out_pos = out_block; out_pos < block_end; out_pos++, in_pos += fscale) {
        int i0, i1;
        std::tie(i0, i1) = window.input_range(in_pos);
        if (i0 + in_block_i < 0)
          i0 = -in_block_i;
        if (i1 + in_block_i >= n_in)
          i1 = n_in - 1 - in_block_i;
        float f = 0;
        int i = i0;

#ifdef __SSE2__
        __m128 f4 = _mm_setzero_ps();
        __m128 x4 = _mm_setr_ps(i - in_pos, i+1 - in_pos, i+2 - in_pos, i+3 - in_pos);
        for (; i + 3 <= i1; i += 4) {
          __m128 w4 = window(x4);

          f4 = _mm_add_ps(f4, _mm_mul_ps(_mm_loadu_ps(in_block_ptr + i), w4));
          x4 = _mm_add_ps(x4, _mm_set1_ps(4));
        }

        f4 = _mm_add_ps(f4, _mm_shuffle_ps(f4, f4, _MM_SHUFFLE(1, 0, 3, 2)));
        f4 = _mm_add_ps(f4, _mm_shuffle_ps(f4, f4, _MM_SHUFFLE(0, 1, 0, 1)));
        f = _mm_cvtss_f32(f4);
#endif

        float x = i - in_pos;
        for (; i <= i1; i++, x++) {
          float w = window(x);
          f += in_block_ptr[i] * w;
        }
        assert(out_pos >= out_begin && out_pos < out_end);
        out[out_pos] = ConvertSatNorm<Out>(f);
      }
    }
  }


  /**
   * @brief Resample multi-channel signal and convert to Out
   *
   * Calculates a range of resampled signal.
   * The function can seamlessly resample the input and produce the result in chunks.
   * To reuse memory and still simulate chunk processing, adjust the in/out pointers.
   *
   * @tparam static_channels   number of channels, if known at compile time, or -1
   */
  template <int static_channels, typename Out>
  void Resample(
        Out *__restrict__ out, int64_t out_begin, int64_t out_end, double out_rate,
        const float *__restrict__ in, int64_t n_in, double in_rate,
        int dynamic_num_channels) {
    static_assert(static_channels != 0, "Static number of channels must be positive (use static) "
                                        "or negative (use dynamic).");
    assert(out_rate > 0 && in_rate > 0 && "Sampling rate must be positive");
    if (dynamic_num_channels == 1) {
      // fast path
      Resample(out, out_begin, out_end, out_rate, in, n_in, in_rate);
      return;
    }
    // the check below is compile time, so num_channels will be a compile-time constant
    // or a run-time constant, depending on the value of static_channels
    const int num_channels = static_channels < 0 ? dynamic_num_channels : static_channels;
    assert(num_channels > 0);

    int64_t in_pos = 0;
    int64_t block = 1 << 10;  // still leaves 13 significant bits for fractional part
    double scale = in_rate / out_rate;
    float fscale = scale;
    SmallVector<float, (static_channels < 0 ? 16 : static_channels)> tmp;
    tmp.resize(num_channels);
    for (int64_t out_block = out_begin; out_block < out_end; out_block += block) {
      int64_t block_end = std::min(out_block + block, out_end);
      double in_block_f = out_block * scale;
      int64_t in_block_i = std::floor(in_block_f);
      float in_pos = in_block_f - in_block_i;
      const float *__restrict__ in_block_ptr = in + in_block_i * num_channels;
      for (int64_t out_pos = out_block; out_pos < block_end; out_pos++, in_pos += fscale) {
        int i0, i1;
        std::tie(i0, i1) = window.input_range(in_pos);
        if (i0 + in_block_i < 0)
          i0 = -in_block_i;
        if (i1 + in_block_i >= n_in)
          i1 = n_in - 1 - in_block_i;

        for (int c = 0; c < num_channels; c++)
          tmp[c] = 0;

        float x = i0 - in_pos;
        int ofs0 = i0 * num_channels;
        int ofs1 = i1 * num_channels;
        for (int in_ofs = ofs0; in_ofs <= ofs1; in_ofs += num_channels, x++) {
          float w = window(x);
          for (int c = 0; c < num_channels; c++) {
            assert(in_block_ptr + in_ofs + c >= in &&
                   in_block_ptr + in_ofs + c < in + n_in * num_channels);
            tmp[c] += in_block_ptr[in_ofs + c] * w;
          }
        }
        assert(out_pos >= out_begin && out_pos < out_end);
        for (int c = 0; c < num_channels; c++)
          out[out_pos * num_channels + c] = ConvertSatNorm<Out>(tmp[c]);
      }
    }
  }


  /**
   * @brief Resample multi-channel signal and convert to Out
   *
   * Calculates a range of resampled signal.
   * The function can seamlessly resample the input and produce the result in chunks.
   * To reuse memory and still simulate chunk processing, adjust the in/out pointers.
   */
  template <typename Out>
  void Resample(
        Out *__restrict__ out, int64_t out_begin, int64_t out_end, double out_rate,
        const float *__restrict__ in, int64_t n_in, double in_rate,
        int num_channels) {
    VALUE_SWITCH(num_channels, static_channels, (1, 2, 3, 4, 5, 6, 7, 8),
      (Resample<static_channels, Out>(out, out_begin, out_end, out_rate,
        in, n_in, in_rate, static_channels);),
      (Resample<-1, Out>(out, out_begin, out_end, out_rate,
        in, n_in, in_rate, num_channels)));
  }
};

}  // namespace resampling
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_RESAMPLING_H_
