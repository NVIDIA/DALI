// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/signal/resampling_cpu.h"

#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include <cmath>
#include <functional>
#include <utility>
#include <vector>
#include "dali/core/convert.h"
#include "dali/core/math_util.h"
#include "dali/core/small_vector.h"
#include "dali/core/static_switch.h"

namespace dali {
namespace kernels {
namespace signal {

namespace resampling {

#if defined(__ARM_NEON)

inline float32x4_t evaluate(const ResamplingWindow &window, float32x4_t x) {
  float32x4_t fi = vfmaq_n_f32(vdupq_n_f32(window.center), x, window.scale);
  int32x4_t i = vcvtq_s32_f32(fi);
  float32x4_t fifloor = vcvtq_f32_s32(i);
  float32x4_t di = vsubq_f32(fi, fifloor);
  int idx[4] = {vgetq_lane_s32(i, 0), vgetq_lane_s32(i, 1), vgetq_lane_s32(i, 2),
                vgetq_lane_s32(i, 3)};
  float32x2_t c0 = vld1_f32(&window.lookup[idx[0]]);
  float32x2_t c1 = vld1_f32(&window.lookup[idx[1]]);
  float32x2_t c2 = vld1_f32(&window.lookup[idx[2]]);
  float32x2_t c3 = vld1_f32(&window.lookup[idx[3]]);
  float32x4x2_t w = vuzpq_f32(vcombine_f32(c0, c1), vcombine_f32(c2, c3));
  float32x4_t curr = w.val[0];
  float32x4_t next = w.val[1];
  return vfmaq_f32(curr, di, vsubq_f32(next, curr));
}

inline float32x4_t vsetq_f32(float x0, float x1, float x2, float x3) {
  float32x4_t x;
  x = vdupq_n_f32(x0);
  x = vsetq_lane_f32(x1, x, 1);
  x = vsetq_lane_f32(x2, x, 2);
  x = vsetq_lane_f32(x3, x, 3);
  return x;
}

inline float filter_vec(const ResamplingWindow &window, int &i_ref, float in_pos, int i1,
                        const float *in) {
  const float32x4_t _0123 = vsetq_f32(0, 1, 2, 3);
  float32x4_t f4 = vdupq_n_f32(0);

  int i = i_ref;
  float32x4_t x4 = vaddq_f32(vdupq_n_f32(i - in_pos), _0123);

  for (; i + 3 < i1; i += 4) {
    float32x4_t w4 = evaluate(window, x4);
    f4 = vfmaq_f32(f4, vld1q_f32(in + i), w4);
    x4 = vaddq_f32(x4, vdupq_n_f32(4));
  }
  // Sum elements in f4
  float32x2_t f2 = vpadd_f32(vget_low_f32(f4), vget_high_f32(f4));
  f2 = vpadd_f32(f2, f2);
  i_ref = i;
  return vget_lane_f32(f2, 0);
}

#elif defined(__SSE2__)

inline __m128 evaluate(const ResamplingWindow &window, __m128 x) {
  __m128 fi = _mm_add_ps(x * _mm_set1_ps(window.scale), _mm_set1_ps(window.center));
  __m128i i = _mm_cvttps_epi32(fi);
  __m128 fifloor = _mm_cvtepi32_ps(i);
  __m128 di = _mm_sub_ps(fi, fifloor);
  int idx[4];
  _mm_storeu_si128(reinterpret_cast<__m128i *>(idx), i);
  __m128 curr = _mm_setr_ps(window.lookup[idx[0]], window.lookup[idx[1]],
                            window.lookup[idx[2]], window.lookup[idx[3]]);
  __m128 next = _mm_setr_ps(window.lookup[idx[0] + 1], window.lookup[idx[1] + 1],
                            window.lookup[idx[2] + 1], window.lookup[idx[3] + 1]);
  return _mm_add_ps(curr, _mm_mul_ps(di, _mm_sub_ps(next, curr)));
}

inline float filter_vec(const ResamplingWindow &window, int &i_ref, float in_pos, int i1,
                        const float *in) {
  __m128 f4 = _mm_setzero_ps();
  int i = i_ref;
  __m128 x4 = _mm_setr_ps(i - in_pos, i + 1 - in_pos, i + 2 - in_pos, i + 3 - in_pos);
  for (; i + 3 < i1; i += 4) {
    __m128 w4 = evaluate(window, x4);

    f4 = _mm_add_ps(f4, _mm_mul_ps(_mm_loadu_ps(in + i), w4));
    x4 = _mm_add_ps(x4, _mm_set1_ps(4));
  }
  i_ref = i;

  // Sum elements in f4
  f4 = _mm_add_ps(f4, _mm_shuffle_ps(f4, f4, _MM_SHUFFLE(1, 0, 3, 2)));
  f4 = _mm_add_ps(f4, _mm_shuffle_ps(f4, f4, _MM_SHUFFLE(0, 1, 0, 1)));
  return _mm_cvtss_f32(f4);
}

#else

inline float filter_vec(const ResamplingWindow &, int &, float, int, const float *) {
  return 0;
}

#endif

/**
 * @brief Resample single-channel signal and convert to Out
 *
 * Calculates a range of resampled signal.
 * The function can seamlessly resample the input and produce the result in chunks.
 * To reuse memory and still simulate chunk processing, adjust the in/out pointers.
 */
template <typename Out>
void ResampleCPUImpl(ResamplingWindow window, Out *__restrict__ out, int64_t out_begin,
                     int64_t out_end, double out_rate, const float *__restrict__ in, int64_t n_in,
                     double in_rate) {
  assert(out_rate > 0 && in_rate > 0 && "Sampling rate must be positive");
  int64_t block = 1 << 8;  // still leaves 15 significant bits for fractional part
  double scale = in_rate / out_rate;
  float fscale = scale;

  for (int64_t out_block = out_begin; out_block < out_end; out_block += block) {
    int64_t block_end = std::min(out_block + block, out_end);
    double in_block_f = out_block * scale;
    int64_t in_block_i = std::floor(in_block_f);
    float in_pos = in_block_f - in_block_i;
    const float *__restrict__ in_block_ptr = in + in_block_i;
    for (int64_t out_pos = out_block; out_pos < block_end; out_pos++, in_pos += fscale) {
      auto irange = window.input_range(in_pos);
      int i0 = irange.i0;
      int i1 = irange.i1;
      if (i0 + in_block_i < 0)
        i0 = -in_block_i;
      if (i1 + in_block_i > n_in)
        i1 = n_in - in_block_i;
      int i = i0;

      float f = filter_vec(window, i, in_pos, i1, in_block_ptr);

      float x = i - in_pos;
      for (; i < i1; i++, x++) {
        float w = window(x);
        f += in_block_ptr[i] * w;
      }
      assert(out_pos >= out_begin && out_pos < out_end);
      auto rel_pos = out_pos - out_begin;
      out[rel_pos] = ConvertSatNorm<Out>(f);
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
void ResampleCPUImpl(ResamplingWindow window, Out *__restrict__ out, int64_t out_begin,
                     int64_t out_end, double out_rate, const float *__restrict__ in, int64_t n_in,
                     double in_rate, int dynamic_num_channels) {
  static_assert(static_channels != 0,
                "Static number of channels must be positive (use static) "
                "or negative (use dynamic).");
  assert(out_rate > 0 && in_rate > 0 && "Sampling rate must be positive");
  if (dynamic_num_channels == 1) {
    // fast path
    ResampleCPUImpl(window, out, out_begin, out_end, out_rate, in, n_in, in_rate);
    return;
  }
  // the check below is compile time, so num_channels will be a compile-time constant
  // or a run-time constant, depending on the value of static_channels
  const int num_channels = static_channels < 0 ? dynamic_num_channels : static_channels;
  assert(num_channels > 0);

  int64_t block = 1 << 8;  // still leaves 15 significant bits for fractional part
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
      auto irange = window.input_range(in_pos);
      int i0 = irange.i0;
      int i1 = irange.i1;
      if (i0 + in_block_i < 0)
        i0 = -in_block_i;
      if (i1 + in_block_i > n_in)
        i1 = n_in - in_block_i;

      for (int c = 0; c < num_channels; c++)
        tmp[c] = 0;

      float x = i0 - in_pos;
      int ofs0 = i0 * num_channels;
      int ofs1 = i1 * num_channels;
      for (int in_ofs = ofs0; in_ofs < ofs1; in_ofs += num_channels, x++) {
        float w = window(x);
        for (int c = 0; c < num_channels; c++) {
          assert(in_block_ptr + in_ofs + c >= in &&
                 in_block_ptr + in_ofs + c < in + n_in * num_channels);
          tmp[c] += in_block_ptr[in_ofs + c] * w;
        }
      }
      assert(out_pos >= out_begin && out_pos < out_end);
      auto rel_pos = out_pos - out_begin;
      for (int c = 0; c < num_channels; c++)
        out[rel_pos * num_channels + c] = ConvertSatNorm<Out>(tmp[c]);
    }
  }
}

/**
 * @brief Resample multi-channel (or single channel) signal and convert to Out
 *
 * Calculates a range of resampled signal.
 * The function can resample a region-of-interest (ROI) of the output, specified by `out_begin` and
 * `out_end`. In this case, the output pointer points to the beginning of the ROI.
 */
template <typename Out>
void ResampleCPUImpl(ResamplingWindow window, Out *__restrict__ out, int64_t out_begin,
                     int64_t out_end, double out_rate, const float *__restrict__ in, int64_t n_in,
                     double in_rate, int num_channels) {
  VALUE_SWITCH(num_channels, static_channels, (1, 2, 3, 4, 5, 6, 7, 8),
    (ResampleCPUImpl<static_channels, Out>(window, out, out_begin, out_end, out_rate,
      in, n_in, in_rate, static_channels);),
    (ResampleCPUImpl<-1, Out>(window, out, out_begin, out_end, out_rate,
      in, n_in, in_rate, num_channels)));
}

#define DALI_INSTANTIATE_RESAMPLER_CPU_OUT(Out)                                             \
  template void ResampleCPUImpl(ResamplingWindow window, Out *__restrict__ out,             \
                                int64_t out_begin, int64_t out_end, double out_rate,        \
                                const float *__restrict__ in, int64_t n_in, double in_rate, \
                                int num_channels);

#define DALI_INSTANTIATE_RESAMPLER_CPU()        \
  DALI_INSTANTIATE_RESAMPLER_CPU_OUT(float);    \
  DALI_INSTANTIATE_RESAMPLER_CPU_OUT(int8_t);   \
  DALI_INSTANTIATE_RESAMPLER_CPU_OUT(uint8_t);  \
  DALI_INSTANTIATE_RESAMPLER_CPU_OUT(int16_t);  \
  DALI_INSTANTIATE_RESAMPLER_CPU_OUT(uint16_t); \
  DALI_INSTANTIATE_RESAMPLER_CPU_OUT(int32_t);  \
  DALI_INSTANTIATE_RESAMPLER_CPU_OUT(uint32_t);

DALI_INSTANTIATE_RESAMPLER_CPU();


}  // namespace resampling
}  // namespace signal
}  // namespace kernels
}  // namespace dali
