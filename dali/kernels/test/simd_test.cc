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

#include <gtest/gtest.h>
#include "dali/kernels/common/simd.h"
#include "dali/core/convert.h"

namespace dali {
namespace kernels {
namespace simd {
namespace test {

#ifdef __SSE2__

template <typename Out, int nvec = sizeof(float)/sizeof(Out)>
void TestConvertStore(float4x<nvec> vec) {
  Out out[nvec * 4];  // NOLINT
  store_f(out, vec);
  float flt[nvec * 4];  // NOLINT
  for (int i = 0; i < nvec; i++)
    _mm_storeu_ps(flt + i * 4, vec.v[i]);
  for (int i = 0; i < nvec * 4; i++)
    EXPECT_EQ(out[i], ConvertSat<Out>(flt[i]));
}

template <typename Out, int nvec = sizeof(int32_t)/sizeof(Out)>
void TestConvertStore(i128x<nvec> vec) {
  Out out[nvec * 4];  // NOLINT
  store_i32(out, vec);
  int32_t i32[nvec * 4];  // NOLINT
  for (int i = 0; i < nvec; i++)
    _mm_storeu_si128(reinterpret_cast<__m128i*>(i32 + i * 4), vec.v[i]);
  for (int i = 0; i < nvec * 4; i++)
    EXPECT_EQ(out[i], ConvertSat<Out>(i32[i]));
}

template <typename In, int nvec = sizeof(float)/sizeof(In)>
void TestConvertLoad(In lo, In hi) {
  int lanes = 16 / sizeof(In);
  In in[lanes];  // NOLINT
  // promote to int64_t computation to avoid integer overflow
  for (int64_t i = 0; i < lanes; i++) {
    in[i] = lo + (hi - lo) * i / (lanes - 1);
  }
  float4x<nvec> v = load_f(in);
  for (int i = 0; i < nvec; i++) {
    float tmp[4];
    _mm_storeu_ps(tmp, v.v[i]);
    for (int j = 0; j < 4; j++) {
      float ref = in[4*i + j];
      EXPECT_EQ(tmp[j], ref);
    }
  }
}

template <int n>
float4x<n> make_vec_f(float min, float max) {
  float range = max - min;
  min -= range / 4;
  max += range / 4;
  range = max - min;
  float delta = range / (4 * n - 1);
  float value = min;
  float4x<n> out;
  for (int i = 0; i < n; i++) {
    float tmp[4];
    for (int j = 0; j < 4; j++) {
      tmp[j] = value;
      value += delta;
    }
    out.v[i] = _mm_loadu_ps(tmp);
  }
  return out;
}

template <int n>
i128x<n> make_vec_i32(float min, float max) {
  float range = max - min;
  min -= range / 4;
  max += range / 4;
  range = max - min;
  float delta = range / (4 * n - 1);
  float value = min;
  i128x<n> out;
  for (int i = 0; i < n; i++) {
    int32_t tmp[4];
    for (int j = 0; j < 4; j++) {
      tmp[j] = value;
      value += delta;
    }
    out.v[i] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tmp));
  }
  return out;
}

TEST(SSE2Test, ConvertStore) {
  TestConvertStore<int8_t>(make_vec_f<4>(-128, 127));
  TestConvertStore<uint8_t>(make_vec_f<4>(0, 255));
  TestConvertStore<int16_t>(make_vec_f<2>(-32768, 32767));
  TestConvertStore<uint16_t>(make_vec_f<2>(0, 65535));
  TestConvertStore<int32_t>(make_vec_f<1>(-1000000, 1000000));
  TestConvertStore<int32_t>(make_vec_f<1>(-2.5e+9, 2.5e+9));

  TestConvertStore<int8_t>(make_vec_i32<4>(-128, 127));
  TestConvertStore<uint8_t>(make_vec_i32<4>(0, 255));
  TestConvertStore<int16_t>(make_vec_i32<2>(-32768, 32767));
  TestConvertStore<uint16_t>(make_vec_i32<2>(0, 65535));
  TestConvertStore<int32_t>(make_vec_i32<1>(-1000000, 1000000));
}

TEST(SSE2Test, ConvertLoad) {
  TestConvertLoad<int8_t>(-128, 127);
  TestConvertLoad<int16_t>(-32768, 32767);
  TestConvertLoad<uint8_t>(0, 255);
  TestConvertLoad<uint16_t>(0, 65535);
  TestConvertLoad<int32_t>(-1000000000, 1000000000);
}

#endif  // __SSE2__

}  // namespace test
}  // namespace simd
}  // namespace kernels
}  // namespace dali
