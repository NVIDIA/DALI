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

inline __m128i clamp_round(__m128 f, float lo, float hi) {
  f = _mm_min_ps(_mm_max_ps(f, _mm_set1_ps(lo)), _mm_set1_ps(hi));
  return _mm_cvtps_epi32(f);  // round
}

inline void store(int8_t *i8, i128x4 iv) {
  __m128i sv0 = _mm_packs_epi32(iv.v[0], iv.v[1]);
  __m128i sv1 = _mm_packs_epi32(iv.v[2], iv.v[3]);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(i8), _mm_packs_epi16(sv0, sv1));
}

inline void store(uint8_t *u8, i128x4 iv) {
  __m128i sv0 = _mm_packs_epi32(iv.v[0], iv.v[1]);
  __m128i sv1 = _mm_packs_epi32(iv.v[2], iv.v[3]);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(u8), _mm_packus_epi16(sv0, sv1));
}

inline void store(int16_t *i16, i128x2 iv) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(i16), _mm_packs_epi32(iv.v[0], iv.v[1]));
}

inline void store(uint16_t *u16, i128x2 iv) {
  __m128i sh = _mm_set_epi32(0, 0, 0, 1);
  __m128i one = _mm_set1_epi32(1);
  // avoid signed saturation by shifting right and saving LSB, then reconsitituting it
  __m128i hi0 = _mm_srl_epi32(iv.v[0], sh);
  __m128i hi1 = _mm_srl_epi32(iv.v[1], sh);
  __m128i lo0 = _mm_and_si128(iv.v[0], one);
  __m128i lo1 = _mm_and_si128(iv.v[0], one);
  __m128i lo = _mm_packs_epi32(lo0, lo1);
  __m128i hi = _mm_packs_epi32(hi0, hi1);
  __m128i out = _mm_or_si128(_mm_sll_epi16(hi, sh), lo);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(u16), out);
}

inline float4x4 load_f(const uint8_t *u8) {
  __m128i in = _mm_loadu_si128(reinterpret_cast<const __m128i *>(u8));
  __m128i zero = _mm_setzero_si128();
  __m128i lo16 = _mm_unpacklo_epi8(in, zero);
  __m128i hi16 = _mm_unpackhi_epi8(in, zero);
  __m128i i32_0 = _mm_unpacklo_epi8(lo16, zero);
  __m128i i32_1 = _mm_unpackhi_epi8(lo16, zero);
  __m128i i32_2 = _mm_unpacklo_epi8(hi16, zero);
  __m128i i32_3 = _mm_unpackhi_epi8(hi16, zero);
  return {{ _mm_cvtepi32_ps(i32_0),
            _mm_cvtepi32_ps(i32_1),
            _mm_cvtepi32_ps(i32_2),
            _mm_cvtepi32_ps(i32_3) }};
}

inline __m128i sext8_32(__m128i i) {
  __m128i sh = _mm_set_epi32(0, 0, 0, 24);
  return _mm_sra_epi32(_mm_sll_epi32(i, sh), sh);
}

inline float4x4 load_f(const int8_t *i8) {
  __m128i in = _mm_loadu_si128(reinterpret_cast<const __m128i *>(i8));
  __m128i zero = _mm_setzero_si128();
  __m128i lo16 = _mm_unpacklo_epi8(in, zero);
  __m128i hi16 = _mm_unpackhi_epi8(in, zero);
  __m128i i32_0 = _mm_unpacklo_epi8(lo16, zero);
  __m128i i32_1 = _mm_unpackhi_epi8(lo16, zero);
  __m128i i32_2 = _mm_unpacklo_epi8(hi16, zero);
  __m128i i32_3 = _mm_unpackhi_epi8(hi16, zero);
  return {{ _mm_cvtepi32_ps(sext8_32(i32_0)),
            _mm_cvtepi32_ps(sext8_32(i32_1)),
            _mm_cvtepi32_ps(sext8_32(i32_2)),
            _mm_cvtepi32_ps(sext8_32(i32_3)) }};
}

inline __m128i sext16_32(__m128i i) {
  __m128i sh = _mm_set_epi32(0, 0, 0, 16);
  return _mm_sra_epi32(_mm_sll_epi32(i, sh), sh);
}

inline float4x2 load_f(const uint16_t *u16) {
  __m128i in = _mm_loadu_si128(reinterpret_cast<const __m128i *>(u16));
  __m128i zero = _mm_setzero_si128();
  __m128i i32_0 = _mm_unpacklo_epi16(in, zero);
  __m128i i32_1 = _mm_unpackhi_epi16(in, zero);
  return {{ _mm_cvtepi32_ps(i32_0), _mm_cvtepi32_ps(i32_1) }};
}

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

template <typename Out>
inline std::enable_if_t<std::is_integral<Out>::value>
store(Out *out, float4x<sizeof(float)/sizeof(Out)> f) {
  constexpr int nvec = sizeof(float)/sizeof(Out);
  constexpr float lo = min_value<Out>();
  constexpr float hi = max_value<Out>();
  i128x<nvec> iv;
  for (int i = 0; i < nvec; i++)
    iv.v[i] = clamp_round(f.v[i], lo, hi);
  store(out, iv);
}

inline void store(float *out, float4x1 f) {
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
    multivec m;
    for (int i = 0; i < num_vecs; i += load_vecs) {
      auto tmp = simd::load_f(in + i * load_lanes);
      for (int j = 0; j < load_vecs; j++)
        m.v[i + j] = tmp.v[j];
    }
    return m;
  }
};


template <int num_vecs, typename Out>
DALI_FORCEINLINE static void store(Out *out, multivec<num_vecs> m) noexcept {
  constexpr int store_lanes = 16 / sizeof(Out);
  constexpr int store_vecs = store_lanes / 4;
  for (int i = 0; i < num_vecs; i += store_vecs) {
    float4x<store_vecs> slice;
    for (int j = 0; j < store_vecs; j++)
      slice.v[j] = m.v[i + j];
    simd::store(out + i * store_lanes, slice);
  }
}

#endif  // __SSE2__

}  // namespace simd
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_SIMD_H_
