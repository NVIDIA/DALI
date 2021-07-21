// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include <vector>
#include <cstring>
#include <typeinfo>
#include "dali/core/float16.h"
#include "dali/test/device_test.h"
#include "dali/core/dev_buffer.h"

namespace dali {

__device__ auto dev_to_string(float16 x) {
  return dev_to_string(static_cast<float>(x));
}

TEST(FP16Host, Construction) {
#ifndef __CUDA_ARCH__
  float16 a = {};
  EXPECT_EQ(a.impl, float16::impl_t());
#endif
  float16 b = 42;
  EXPECT_EQ(static_cast<float>(b), 42.0f);
  float16 c{-42L};
  EXPECT_EQ(static_cast<float>(c), -42);
  long long ll = 1234;  // NOLINT
  float16 d(ll);
  EXPECT_EQ(static_cast<float>(d), 1234);
  float16 e = { -5.5f };
  EXPECT_EQ(static_cast<float>(e), -5.5f);
  float16 f = 2u;
  EXPECT_EQ(static_cast<float>(f), 2.0f);
#ifndef __CUDA_ARCH__
  float16 g = f.impl;
  EXPECT_EQ(0, std::memcmp(&g, &f, 2));
#endif
}

DEVICE_TEST(FP16Dev, Construction, 1, 1) {
  float16 a = {};
  DEV_EXPECT_EQ(static_cast<float>(a), 0.0f);
  float16 b = 42;
  DEV_EXPECT_EQ(static_cast<float>(b), 42.0f);
  float16 c{-42L};
  DEV_EXPECT_EQ(static_cast<float>(c), -42);
  long long ll = 1234;  // NOLINT
  float16 d(ll);
  DEV_EXPECT_EQ(static_cast<float>(d), 1234);
  float16 e = { -5.5f };
  DEV_EXPECT_EQ(static_cast<float>(e), -5.5f);
  float16 f = 2u;
  DEV_EXPECT_EQ(static_cast<float>(f), 2.0f);
#ifdef __CUDA_ARCH__
  float16 g{f.impl};
  DEV_EXPECT_EQ(static_cast<float>(g), 2.0f);
#endif
}

TEST(FP16Host, Conversions) {
  float16 a = {};
#ifndef __CUDA_ARCH__
  EXPECT_EQ(a.impl, float16::impl_t());
#endif
  a = 42;
  EXPECT_EQ(static_cast<float>(a), 42.0f);
  a = -42L;
  EXPECT_EQ(static_cast<float>(a), -42);
  long long ll = 1234;  // NOLINT
  a = ll;
  EXPECT_EQ(static_cast<float>(a), 1234);
  a = -5.5f;
  EXPECT_EQ(static_cast<float>(a), -5.5f);
  a = 2u;
  EXPECT_EQ(static_cast<float>(a), 2.0f);
}

DEVICE_TEST(FP16Dev, Conversions, 1, 1) {
  float16 a = {};
#ifdef __CUDA_ARCH__
  DEV_EXPECT_EQ(static_cast<float>(a.impl), 0.0f);
#endif
  a = 42;
  DEV_EXPECT_EQ(static_cast<float>(a), 42.0f);
  long long ll = 1234;  // NOLINT
  a = ll;
  DEV_EXPECT_EQ(static_cast<float>(a), 1234);
  a = -5.5f;
  DEV_EXPECT_EQ(static_cast<float>(a), -5.5f);
  a = 2u;
  DEV_EXPECT_EQ(static_cast<float>(a), 2.0f);
}

TEST(FP16Host, Arithm) {
  float16 a = 5;
  float16 b = 42;
  float16 c = a + b;
  EXPECT_EQ(static_cast<float>(c), 47);
  c = a - b;
  EXPECT_EQ(static_cast<float>(c), -37);
  b = -b;
  EXPECT_EQ(static_cast<float>(b), -42);
  b = +b;
  EXPECT_EQ(static_cast<float>(b), -42);
  c = a * b;
  EXPECT_EQ(static_cast<float>(c), -210);
  a = 12;
  b = 4;
  c = a / b;
  EXPECT_EQ(static_cast<float>(c), 3);
}

DEVICE_TEST(FP16Dev, Arithm, 1, 1) {
  float16 a = 5;
  float16 b = 42;
  float16 c = a + b;
  DEV_EXPECT_EQ(static_cast<float>(c), 47);
  c = a - b;
  DEV_EXPECT_EQ(static_cast<float>(c), -37);
  b = -b;
  DEV_EXPECT_EQ(static_cast<float>(b), -42);
  b = +b;
  DEV_EXPECT_EQ(static_cast<float>(b), -42);
  c = a * b;
  DEV_EXPECT_EQ(static_cast<float>(c), -210);
  a = 12;
  b = 4;
  c = a / b;
  DEV_EXPECT_EQ(static_cast<float>(c), 3);
}

#define EXPECT_FP16_EQ(a, b) EXPECT_EQ(float(float16(a)), float(b))  // NOLINT

TEST(FP16Host, ArithmMixed) {
  float16 a = 2048;
  EXPECT_FP16_EQ(a - 4095, -2048);  // rounding
  EXPECT_FP16_EQ(4095 - a, 2048);  // rounding
  EXPECT_EQ(a + (-4095.0f), -2047.0f);
  EXPECT_EQ((-4095.0f) + a, -2047.0f);
  a = 1234;
  EXPECT_FP16_EQ(a * 128u, float16(1234 * 128.0f));  // infinity
  EXPECT_FP16_EQ(a / 128u, 1234.0f / 128.0f);
  EXPECT_EQ(128.0 * a, float16(1234 * 128.0));  // infinity
  EXPECT_EQ(a / 128.0, 1234.0 / 128.0);
}

#define DEV_EXPECT_FP16_EQ(a, b) DEV_EXPECT_EQ(float(float16(a)), float(b))  // NOLINT

DEVICE_TEST(FP16Dev, ArithmMixed, 1, 1) {
  float16 a = 2048;
  DEV_EXPECT_FP16_EQ(a - 4095, -2048);  // rounding
  DEV_EXPECT_FP16_EQ(4095 - a, 2048);  // rounding
  DEV_EXPECT_EQ(a + (-4095.0f), -2047);
  DEV_EXPECT_EQ((-4095.0f) + a, -2047);
  a = 1234;
  DEV_EXPECT_FP16_EQ(a * 128u, float16(1234 * 128.0f));  // infinity
  DEV_EXPECT_FP16_EQ(a / 128u, 1234.0f / 128.0f);
  DEV_EXPECT_EQ(128.0 * a, float16(1234 * 128.0));  // infinity
  DEV_EXPECT_EQ(a / 128.0, 1234.0 / 128.0);
}

TEST(FP16Host, Compound) {
  float16 a = 5;
  float16 b = 42;
  a += b;
  EXPECT_EQ(static_cast<float>(a), 47);
  a = 5;
  a -= b;
  EXPECT_EQ(static_cast<float>(a), -37);
  a = 5;
  a *= b;
  EXPECT_EQ(static_cast<float>(a), 210);
  a = 12;
  b = 4;
  a /= b;
  EXPECT_EQ(static_cast<float>(a), 3);

  a = 5;
  a += 42.0;
  EXPECT_EQ(static_cast<float>(a), 47);
  a = 2048;
  a -= 4095;
  EXPECT_EQ(static_cast<float>(a), -2048);  // rounding
  a = 5;
  a -= 42.0f;
  EXPECT_EQ(static_cast<float>(a), -37);
  a = 5;
  a *= 42u;
  EXPECT_EQ(static_cast<float>(a), 210);
  a = 12;
  a /= '\4';
  EXPECT_EQ(static_cast<float>(a), 3);
}

DEVICE_TEST(FP16Dev, Compound, 1, 1) {
  float16 a = 5;
  float16 b = 42;
  a += b;
  DEV_EXPECT_EQ(static_cast<float>(a), 47);
  a = 5;
  a -= b;
  DEV_EXPECT_EQ(static_cast<float>(a), -37);
  a = 5;
  a *= b;
  DEV_EXPECT_EQ(static_cast<float>(a), 210);
  a = 12;
  b = 4;
  a /= b;
  DEV_EXPECT_EQ(static_cast<float>(a), 3);

  a = 5;
  a += 42.0;
  DEV_EXPECT_EQ(static_cast<float>(a), 47);
  a = 2048;
  a -= 4095;
  DEV_EXPECT_EQ(static_cast<float>(a), -2048);  // rounding
  a = 5;
  a -= 42.0f;
  DEV_EXPECT_EQ(static_cast<float>(a), -37);
  a = 5;
  a *= 42u;
  DEV_EXPECT_EQ(static_cast<float>(a), 210);
  a = 12;
  a /= '\4';
  DEV_EXPECT_EQ(static_cast<float>(a), 3);
}


TEST(FP16Host, Comparison) {
  auto test_cmp = [&](float16 x, auto y) {
    EXPECT_EQ((x == y), (static_cast<float>(x) == static_cast<double>(y)));
    EXPECT_EQ((y == x), (static_cast<float>(x) == static_cast<double>(y)));
    EXPECT_EQ((x != y), (static_cast<float>(x) != static_cast<double>(y)));
    EXPECT_EQ((y != x), (static_cast<float>(x) != static_cast<double>(y)));

    EXPECT_EQ((x < y), (static_cast<float>(x) < static_cast<double>(y)));
    EXPECT_EQ((y > x), (static_cast<float>(x) < static_cast<double>(y)));

    EXPECT_EQ((x > y), (static_cast<float>(x) > static_cast<double>(y)));
    EXPECT_EQ((y < x), (static_cast<float>(x) > static_cast<double>(y)));

    EXPECT_EQ((x <= y), (static_cast<float>(x) <= static_cast<double>(y)));
    EXPECT_EQ((y >= x), (static_cast<float>(x) <= static_cast<double>(y)));

    EXPECT_EQ((x >= y), (static_cast<float>(x) >= static_cast<double>(y)));
    EXPECT_EQ((y <= x), (static_cast<float>(x) >= static_cast<double>(y)));
  };

  float16 x = 42;
  test_cmp(x, float16(-100));
  test_cmp(x, float16(41));
  test_cmp(x, float16(42));
  test_cmp(x, float16(100));

  test_cmp(x, -100);
  test_cmp(x, 41u);
  test_cmp(x, 41.0f);

  test_cmp(x, 42.0f);
  test_cmp(x, 42);
  test_cmp(x, 42u);
  test_cmp(x, 42.0);

  test_cmp(x, 43.0f);
  test_cmp(x, 430.0);
  test_cmp(x, 100);
  test_cmp(x, 100u);
}

TEST(FP16Host, Literals) {
  auto f1 = 123_hf;
  auto f2 = 1.5_hf;
  static_assert(std::is_same<decltype(f1), float16>::value,
    "The literal should produce a float16.");
  static_assert(std::is_same<decltype(f2), float16>::value,
    "The literal should produce a float16.");
  EXPECT_EQ(static_cast<float>(f1), 123.0f);
  EXPECT_EQ(static_cast<float>(f2), 1.5f);
}

DEVICE_TEST(FP16Dev, Literals, 1, 1) {
  auto f1 = 123_hf;
  auto f2 = 1.5_hf;
  static_assert(std::is_same<decltype(f1), float16>::value,
    "The literal should produce a float16.");
  static_assert(std::is_same<decltype(f2), float16>::value,
    "The literal should produce a float16.");
  DEV_EXPECT_EQ(static_cast<float>(f1), 123.0f);
  DEV_EXPECT_EQ(static_cast<float>(f2), 1.5f);
}

DEVICE_TEST(FP16Dev, Comparison, 1, 1) {
  auto test_cmp = [&](float16 x, auto y) {
    DEV_EXPECT_EQ((x == y), (static_cast<float>(x) == static_cast<double>(y)));
    DEV_EXPECT_EQ((y == x), (static_cast<float>(x) == static_cast<double>(y)));
    DEV_EXPECT_EQ((x != y), (static_cast<float>(x) != static_cast<double>(y)));
    DEV_EXPECT_EQ((y != x), (static_cast<float>(x) != static_cast<double>(y)));

    DEV_EXPECT_EQ((x < y), (static_cast<float>(x) < static_cast<double>(y)));
    DEV_EXPECT_EQ((y > x), (static_cast<float>(x) < static_cast<double>(y)));

    DEV_EXPECT_EQ((x > y), (static_cast<float>(x) > static_cast<double>(y)));
    DEV_EXPECT_EQ((y < x), (static_cast<float>(x) > static_cast<double>(y)));

    DEV_EXPECT_EQ((x <= y), (static_cast<float>(x) <= static_cast<double>(y)));
    DEV_EXPECT_EQ((y >= x), (static_cast<float>(x) <= static_cast<double>(y)));

    DEV_EXPECT_EQ((x >= y), (static_cast<float>(x) >= static_cast<double>(y)));
    DEV_EXPECT_EQ((y <= x), (static_cast<float>(x) >= static_cast<double>(y)));
  };

  float16 x = 42;
  test_cmp(x, float16(-100));
  test_cmp(x, float16(41));
  test_cmp(x, float16(42));
  test_cmp(x, float16(100));

  test_cmp(x, -100);
  test_cmp(x, 41u);
  test_cmp(x, 41.0f);

  test_cmp(x, 42.0f);
  test_cmp(x, 42);
  test_cmp(x, 42u);
  test_cmp(x, 42.0);

  test_cmp(x, 43.0f);
  test_cmp(x, 430.0);
  test_cmp(x, 100);
  test_cmp(x, 100u);
}

template <typename T>
struct TestStruct {
  T *ptr;
  int n;
};

template <typename T>
__global__ void FP16TestKernel(TestStruct<T> ts) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < ts.n)
    ts.ptr[idx] = idx + float16(42);
}

TEST(FP16Linkage, KernelTemplate) {
  DeviceBuffer<float16> buf;
  buf.resize(10*64);
  TestStruct<float16> ts;
  ts.ptr = buf;
  ts.n = buf.size();
  CUDA_CALL(cudaMemset(buf, 0, buf.size_bytes()));
  FP16TestKernel<<<div_ceil(ts.n, 64), 64>>>(ts);
  CUDA_CALL(cudaGetLastError());
  vector<float16> host_buf(buf.size());
  CUDA_CALL(cudaMemcpy(host_buf.data(), buf, buf.size_bytes(), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaDeviceSynchronize());
  for (int i = 0; i < ts.n; i++) {
    EXPECT_EQ(static_cast<float>(host_buf[i]), i + 42.0f);
  }
}

}  // namespace dali
