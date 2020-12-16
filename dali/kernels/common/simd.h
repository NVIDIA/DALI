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

#ifndef DALI_KERNELS_COMMON_SIMD_H_
#define DALI_KERNELS_COMMON_SIMD_H_

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include <cstddef>
#include <limits>
#include <type_traits>
#include "dali/core/force_inline.h"

namespace dali {
namespace kernels {
namespace simd {

#ifdef __SSE2__

template <int n>
struct float4x {
  __m128 v[n];  // NOLINT
};

template <int n>
struct i128x {
  __m128i v[n];  // NOLINT
};

using float4x1 = float4x<1>;
using float4x2 = float4x<2>;
using float4x4 = float4x<4>;
using i128x1 = i128x<1>;
using i128x2 = i128x<2>;
using i128x4 = i128x<4>;

/**
 * @brief Clamp floating point value to range [lo, hi], round to nearest and as int32x4
 */
inline __m128i clamp_round(__m128 f, float lo, float hi) {
  f = _mm_min_ps(_mm_max_ps(f, _mm_set1_ps(lo)), _mm_set1_ps(hi));
  return _mm_cvtps_epi32(f);  // round
}

/**
 * @brief Saturate int32x4x4 to int8x16 and store
 */
inline void store_i32(int8_t *i8, i128x4 iv) {
  __m128i sv0 = _mm_packs_epi32(iv.v[0], iv.v[1]);
  __m128i sv1 = _mm_packs_epi32(iv.v[2], iv.v[3]);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(i8), _mm_packs_epi16(sv0, sv1));
}

/**
 * @brief Saturate int32x4x4 to uint8x16 and store
 */
inline void store_i32(uint8_t *u8, i128x4 iv) {
  __m128i sv0 = _mm_packs_epi32(iv.v[0], iv.v[1]);
  __m128i sv1 = _mm_packs_epi32(iv.v[2], iv.v[3]);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(u8), _mm_packus_epi16(sv0, sv1));
}

/**
 * @brief Saturate and narrow int32x4x2 to int16x8 and store
 */
inline void store_i32(int16_t *i16, i128x2 iv) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(i16), _mm_packs_epi32(iv.v[0], iv.v[1]));
}

inline __m128i clamp_i32_u16(__m128i x) {
  __m128i below = _mm_cmpgt_epi32(_mm_setzero_si128(), x);  // mask is true for negative
  __m128i max_u16 = _mm_set1_epi32(0xffff);
  __m128i above = _mm_cmpgt_epi32(x, max_u16);
  __m128i fill = _mm_and_si128(above, max_u16);   // if x > 0xffff, store 0xffff
  __m128i mask = _mm_or_si128(below, above);      // either below zero or above 0xffff
  __m128i masked = _mm_andnot_si128(mask, x);     // mask out-of-range values
  return _mm_or_si128(masked, fill);              // replace values above 0xffff with 0xffff
}

/**
 * @brief Saturate and narrow int32x4x2 to uint16x8 and store
 */
inline void store_i32(uint16_t *u16, i128x2 iv) {
  __m128i lo = clamp_i32_u16(iv.v[0]);
  __m128i hi = clamp_i32_u16(iv.v[1]);
  __m128i even = _mm_castps_si128(
      _mm_shuffle_ps(_mm_castsi128_ps(lo), _mm_castsi128_ps(hi), _MM_SHUFFLE(2, 0, 2, 0)));
  __m128i odd = _mm_castps_si128(
      _mm_shuffle_ps(_mm_castsi128_ps(lo), _mm_castsi128_ps(hi), _MM_SHUFFLE(3, 1, 3, 1)));

  __m128i out = _mm_or_si128(even, _mm_bslli_si128(odd, 2));  // byte shift

  _mm_storeu_si128(reinterpret_cast<__m128i*>(u16), out);
}

/**
 * @brief Unsafe pack int32x4x2 to uint16x8 and store
 *
 * The result is undefined when values are outside uint16 range
 */
inline void store_i32_unsafe(uint16_t *u16, i128x2 iv) {
  __m128i lo = iv.v[0];
  __m128i hi = iv.v[1];
  __m128i even = _mm_castps_si128(
      _mm_shuffle_ps(_mm_castsi128_ps(lo), _mm_castsi128_ps(hi), _MM_SHUFFLE(2, 0, 2, 0)));
  __m128i odd = _mm_castps_si128(
      _mm_shuffle_ps(_mm_castsi128_ps(lo), _mm_castsi128_ps(hi), _MM_SHUFFLE(3, 1, 3, 1)));

  __m128i out = _mm_or_si128(even, _mm_bslli_si128(odd, 2));  // byte shift

  _mm_storeu_si128(reinterpret_cast<__m128i*>(u16), out);
}

inline void store_i32(int32_t *i32, i128x1 iv) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(i32), iv.v[0]);
}

/**
 * @brief Load int32x4 and convert to 1 float32x4
 */
inline float4x1 load_f(const int32_t *i32) {
  return {{ _mm_cvtepi32_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(i32))) }};
}

/**
 * @brief Load uint8x16 and convert to 4 float32x4
 *
 * Conversion is done by zero-extending to signed int32 and converting to float.
 */
inline float4x4 load_f(const uint8_t *u8) {
  __m128i in = _mm_loadu_si128(reinterpret_cast<const __m128i *>(u8));
  __m128i zero = _mm_setzero_si128();
  __m128i lo16 = _mm_unpacklo_epi8(in, zero);
  __m128i hi16 = _mm_unpackhi_epi8(in, zero);
  __m128i i32_0 = _mm_unpacklo_epi16(lo16, zero);
  __m128i i32_1 = _mm_unpackhi_epi16(lo16, zero);
  __m128i i32_2 = _mm_unpacklo_epi16(hi16, zero);
  __m128i i32_3 = _mm_unpackhi_epi16(hi16, zero);
  return {{ _mm_cvtepi32_ps(i32_0),
            _mm_cvtepi32_ps(i32_1),
            _mm_cvtepi32_ps(i32_2),
            _mm_cvtepi32_ps(i32_3) }};
}

/**
 * @brief Sign-extend 8-bit values stored in LSB of 32-bit lanes
 *
 * Replicate 7th bit in 32-bit lanes to bits 8-31.
 */
inline __m128i sext8_32(__m128i i) {
  __m128i sh = _mm_set_epi32(0, 0, 0, 24);
  return _mm_sra_epi32(_mm_sll_epi32(i, sh), sh);
}


/**
 * @brief Load int8x16 and convert to 4 float32x4
 *
 * Conversion is done by sign-extending to int32 and converting to float.
 */
inline float4x4 load_f(const int8_t *i8) {
  __m128i in = _mm_loadu_si128(reinterpret_cast<const __m128i *>(i8));
  __m128i zero = _mm_setzero_si128();
  __m128i lo16 = _mm_unpacklo_epi8(in, zero);
  __m128i hi16 = _mm_unpackhi_epi8(in, zero);
  __m128i i32_0 = _mm_unpacklo_epi16(lo16, zero);
  __m128i i32_1 = _mm_unpackhi_epi16(lo16, zero);
  __m128i i32_2 = _mm_unpacklo_epi16(hi16, zero);
  __m128i i32_3 = _mm_unpackhi_epi16(hi16, zero);
  return {{ _mm_cvtepi32_ps(sext8_32(i32_0)),
            _mm_cvtepi32_ps(sext8_32(i32_1)),
            _mm_cvtepi32_ps(sext8_32(i32_2)),
            _mm_cvtepi32_ps(sext8_32(i32_3)) }};
}

/**
 * @brief Sign-extend 16-bit values stored in LSB of 32-bit lanes
 *
 * Replicate 15th bit in 32-bit lanes to bits 16-31.
 */
inline __m128i sext16_32(__m128i i) {
  __m128i sh = _mm_set_epi32(0, 0, 0, 16);
  return _mm_sra_epi32(_mm_sll_epi32(i, sh), sh);
}

/**
 * @brief Load uint16x8 and convert to 2 float32x4
 *
 * Conversion is done by zero-extending to signed int32 and converting to float.
 */
inline float4x2 load_f(const uint16_t *u16) {
  __m128i in = _mm_loadu_si128(reinterpret_cast<const __m128i *>(u16));
  __m128i zero = _mm_setzero_si128();
  __m128i i32_0 = _mm_unpacklo_epi16(in, zero);
  __m128i i32_1 = _mm_unpackhi_epi16(in, zero);
  return {{ _mm_cvtepi32_ps(i32_0), _mm_cvtepi32_ps(i32_1) }};
}

/**
 * @brief Load int16x8 and convert to 2 float32x4
 *
 * Conversion is done by sign-extending to signed int32 and converting to float.
 */
inline float4x2 load_f(const int16_t *i16) {
  __m128i in = _mm_loadu_si128(reinterpret_cast<const __m128i *>(i16));
  __m128i zero = _mm_setzero_si128();
  __m128i i32_0 = _mm_unpacklo_epi16(in, zero);
  __m128i i32_1 = _mm_unpackhi_epi16(in, zero);
  return {{ _mm_cvtepi32_ps(sext16_32(i32_0)), _mm_cvtepi32_ps(sext16_32(i32_1)) }};
}

inline float4x1 load_f(const float *f) {
  return {{ _mm_loadu_ps(f) }};
}

/**
 * @brief Converts floating point values to 32-bit signed integers, with proper clamping
 *
 * @remarks NaNs and infinities are stored as -2^31 or 2^31-1, depending on input sign
 */
inline __m128i saturate_f_i32(__m128 f) {
  // this converts f to int32. Out of range values (and NaN) are stored as -2^31
  __m128i raw = _mm_cvtps_epi32(f);
  // xor to check where sign disagrees
  __m128i mask = _mm_xor_si128(_mm_castps_si128(f), raw);
  __m128i adjust = _mm_srli_epi32(mask, 31);  // move the disagreeing sign bit to LSB and subtract
  // this converts 0x80000000 to 0x7fffffff, which is what we want
  return _mm_sub_epi32(raw, adjust);
}

/**
 * @brief Convert multiple vectors of float and convert them to Out.
 */
template <typename Out>
inline std::enable_if_t<std::is_integral<Out>::value>
store_f(Out *out, float4x<sizeof(float)/sizeof(Out)> f) {
  constexpr int nvec = sizeof(float)/sizeof(Out);
  i128x<nvec> iv;
  for (int i = 0; i < nvec; i++)
    iv.v[i] = saturate_f_i32(f.v[i]);
  store_i32(out, iv);
}

/**
 * @brief Convert 2 vectors of float and convert to uint16
 */
inline void store_f(uint16_t *out, float4x2 f) {
  constexpr float lo = 0;
  constexpr float hi = 0xffff;
  i128x<2> iv;
  iv.v[0] = clamp_round(f.v[0], lo, hi);
  iv.v[1] = clamp_round(f.v[1], lo, hi);
  store_i32_unsafe(out, iv);
}

/**
 * @brief Store 1 vector of floats
 */
inline void store_f(float *out, float4x1 f) {
  _mm_storeu_ps(out, f.v[0]);
}

template <int num_vecs>
struct multivec : float4x<num_vecs> {
  DALI_FORCEINLINE static multivec zero() noexcept  {
    multivec m;
    for (int i = 0; i < num_vecs; i++)
      m.v[i] = _mm_setzero_ps();
    return m;
  }

  DALI_FORCEINLINE static multivec load(const float *in) noexcept  {
    multivec m;
    for (int i = 0; i < num_vecs; i++)
      m.v[i] = _mm_loadu_ps(in + 4*i);
    return m;
  }

  template <typename In>
  DALI_FORCEINLINE static multivec load(const In *in) noexcept  {
    constexpr int load_lanes = 16 / sizeof(In);
    constexpr int load_vecs = load_lanes / 4;
    static_assert(load_vecs > 0, "This multivec is too small to be used with this storage type.");
    static_assert(num_vecs * 4 % load_lanes == 0,
      "Total number of lanes is not a multiple of storage lanes.");
    multivec m;
    for (int i = 0; i < num_vecs; i += load_vecs) {
      auto tmp = simd::load_f(in + i * load_lanes);
      for (int j = 0; j < load_vecs; j++)
        m.v[i + j] = tmp.v[j];
    }
    return m;
  }
};

/**
 * @brief Stores multiple vectors of float, converting them to Out
 *
 * The number of lanes must be large enough to fill at least one vector of Out
 */
template <int num_vecs, typename Out>
DALI_FORCEINLINE static void store(Out *out, multivec<num_vecs> m) noexcept {
  constexpr int store_lanes = 16 / sizeof(Out);
  constexpr int store_vecs = store_lanes / 4;
  static_assert(store_vecs > 0, "This multivec is too small to be used with this storage type.");
  static_assert(num_vecs * 4 % store_lanes == 0,
    "Total number of lanes is not a multiple of storage lanes.");
  for (int i = 0; i < num_vecs; i += store_vecs) {
    float4x<store_vecs> slice;
    for (int j = 0; j < store_vecs; j++)
      slice.v[j] = m.v[i + j];
    store_f(out + i * store_lanes, slice);
  }
}

#endif  // __SSE2__

}  // namespace simd
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_SIMD_H_
