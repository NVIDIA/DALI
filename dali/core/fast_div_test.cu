// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/fast_div.h"  // NOLINT
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <iostream>
#include "dali/test/device_test.h"
#include "dali/core/dev_buffer.h"
#include "dali/core/cuda_event.h"

namespace dali {

template <uint32_t divisor>
void TestDiv32(bool quick_test) {
  fast_div<uint32_t> fast = divisor;

  std::cerr << "Testing x / " << divisor << std::endl;
  uint32_t range = quick_test ? (1<<20) : 0xFFFFFFFFu;
  for (uint32_t xx = 0; xx < range; xx++) {
    uint32_t x = quick_test
      ? ((xx >> 12) << 24) | (xx & 0xfff)
      : xx;
    ASSERT_EQ(x / divisor, x / fast) << " when dividing " << x << " / " << divisor;
  }

  uint32_t x = 0xFFFFFFFEu;
  ASSERT_EQ(x / divisor, x / fast) << " when dividing " << x << " / " << divisor;
}

template <uint64_t divisor>
void TestDiv64(bool quick_test) {
  fast_div<uint64_t> fast = divisor;
  std::cerr << "Testing x / " << divisor << std::endl;
  uint64_t range = quick_test ? (1<<20) : 0xFFFFFFFFu;
  for (uint64_t xx = 0; xx < range; xx++) {
    uint64_t x = quick_test
      ? ((xx >> 12) << 56) + (xx & 0xfff)
      : ((xx >> 12) << 44) + (xx & 0xfff);
    ASSERT_EQ(x / divisor, x / fast) << " when dividing " << x << " / " << divisor;
  }
  uint64_t x = 0xFFFFFFFFFFFFFFFE_u64;
  ASSERT_EQ(x / divisor, x / fast) << " when dividing " << x << " / " << divisor;
}

void TestDiv32(bool quick) {
  // use compile-time constants so the compiler can optimize the reference division
  TestDiv32<0xFFFFFFFEu>(quick);
  TestDiv32<3>(quick);
  TestDiv32<5>(quick);
  TestDiv32<7>(quick);
  TestDiv32<1>(quick);
  TestDiv32<0x800000>(quick);
  TestDiv32<19>(quick);
  TestDiv32<42>(quick);
  TestDiv32<12345678>(quick);
  TestDiv32<0x82345678>(quick);
}

void TestDiv64(bool quick) {
  // use compile-time constants so the compiler can optimize the reference division
  TestDiv64<0xFFFFFFFFFFFFFFFE_u64>(quick);
  TestDiv64<3>(quick);
  TestDiv64<5>(quick);
  TestDiv64<7>(quick);
  TestDiv64<1>(quick);
  TestDiv64<0x800000>(quick);
  TestDiv64<0x80000000000000_u64>(quick);
  TestDiv64<19>(quick);
  TestDiv64<42>(quick);
  TestDiv64<12345678901234567_u64>(quick);
  TestDiv64<0x82345678DEADBEEF_u64>(quick);
}

// This test is disabled because it's ridiculously slow - only run when touching fast_div
TEST(FastDiv, DISABLED_U32_Host_Slow) {
  TestDiv32(false);
}

TEST(FastDiv, DISABLED_U64_Host_Slow) {
  TestDiv64(false);
}

// This test is disabled because it's ridiculously slow - only run when touching fast_div
TEST(FastDiv, U32_Host) {
  TestDiv32(true);
}

TEST(FastDiv, U64_Host) {
  TestDiv64(true);
}

// This test is disabled because it's ridiculously slow - only run when touching fast_div
DEVICE_TEST(FastDiv, DISABLED_U32_GPU, dim3(1<<10, 1<<10), (1<<10)) {
  uint32_t start = (blockIdx.x * blockDim.x + threadIdx.x) << 12;
  uint32_t divisor = blockIdx.y + 1;
  fast_div<uint32_t> fast = divisor;
  uint32_t end = start + (1<<12);
  if (end == 0)
    end--;
  for (uint32_t value = start; value < end; value++) {
    DEV_EXPECT_EQ(value / divisor, value / fast);
  }
}

// This test is disabled because it's ridiculously slow - only run when touching fast_div
DEVICE_TEST(FastDiv, DISABLED_U64_GPU_Slow, dim3(1<<10, 1<<10), (1<<10)) {
  uint64_t start = static_cast<uint64_t>(blockIdx.x * blockDim.x + threadIdx.x) << 44;
  uint64_t divisor = blockIdx.y + 1;
  fast_div<uint64_t> fast;
  fast.init(divisor);
  uint64_t end = start + (1<<12);
  if (blockIdx.x == gridDim.x - 1 && threadIdx.x == blockDim.x-1) {
    start = 0xfffffffffffffffe_u64 - (1<<12);
    end = 0xffffffffffffffff_u64;
  }
  for (uint64_t value = start; value < end; value++) {
    auto ref = value / divisor;
    auto result = value / fast;
    if (result != ref) {
      #pragma clang diagnostic push
      #pragma clang diagnostic ignored "-Wformat"
      printf("%llu / %llu   got %llu expected %llu\n", value, divisor, result, ref);
      #pragma clang diagnostic pop
      DEV_ASSERT_EQ(result, ref);
    }
  }
}

DEVICE_TEST(FastDiv, U32_GPU, dim3(1<<10, 11), (1<<10)) {
  static constexpr uint32_t divisors[11] = {
    1, 2, 3, 5, 7, 14, 19, 42,
    0x7fffffffu, 0x80000000u, 0xfffffffeu
  };
  uint32_t start = (blockIdx.x * blockDim.x + threadIdx.x) << 12;
  uint32_t divisor = divisors[blockIdx.y];
  fast_div<uint32_t> fast = divisor;
  uint32_t end = start + (1<<12);
  if (end == 0)
    end--;
  for (uint32_t value = start; value < end; value++) {
    auto ref = value / divisor;
    auto result = value / fast;
    if (result != ref) {
      printf("%u / %u   got %u expected %u\n", value, divisor, result, ref);
      DEV_ASSERT_EQ(value / divisor, value / fast);
    }
  }
}

DEVICE_TEST(FastDiv, U64_GPU, dim3(1<<10, 11), (1<<10)) {
  static constexpr uint64_t divisors[11] = {
    1, 2, 3, 5, 7, 14, 19, 42,
    0x7fffffffffffffff_u64, 0x8000000000000000_u64, 0xfffffffffffffffe_u64
  };
  uint64_t start = static_cast<uint64_t>(blockIdx.x * blockDim.x + threadIdx.x) << 44;
  uint64_t divisor = divisors[blockIdx.y];
  fast_div<uint64_t> fast;
  fast.init(divisor);
  uint64_t end = start + (1<<12);
  if (blockIdx.x == gridDim.x - 1 && threadIdx.x == blockDim.x-1) {
    start = 0xfffffffffffffffe_u64 - (1<<12);
    end = 0xffffffffffffffff_u64;
  }
  for (uint64_t value = start; value < end; value++) {
    auto ref = value / divisor;
    auto result = value / fast;
    if (result != ref) {
      #pragma clang diagnostic push
      #pragma clang diagnostic ignored "-Wformat"
      printf("%llu / %llu   got %llu expected %llu\n", value, divisor, result, ref);
      #pragma clang diagnostic pop
      DEV_ASSERT_EQ(result, ref);
    }
  }
}

template <typename integer>
DALI_HOST_DEV DALI_FORCEINLINE integer div_mod(integer &mod, integer dividend, integer divisor) {
  mod = dividend % divisor;
  return dividend / divisor;
}

template <typename T>
__global__ void NormalDivMod(T *out,
                             T divisor1,
                             T divisor2,
                             T divisor3) {
  T x = static_cast<T>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (int i = 0; i < 6; i++) {
    T q1, r1, q2, r2, q3, r3;
    q1 = div_mod(r1, x, divisor1);
    q2 = div_mod(r2, x, divisor2);
    q3 = div_mod(r3, x, divisor3);
    x = q1 + r1 + q2 + r2 + q3 + r3;
  }
  out[threadIdx.x] += x;
}

template <typename T>
__global__ void FastDivMod(T *out,
                           fast_div<T> divisor1,
                           fast_div<T> divisor2,
                           fast_div<T> divisor3) {
  T x = static_cast<T>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (int i = 0; i < 6; i++) {
    T q1, r1, q2, r2, q3, r3;
    q1 = div_mod(r1, x, divisor1);
    q2 = div_mod(r2, x, divisor2);
    q3 = div_mod(r3, x, divisor3);
    x = q1 + r1 + q2 + r2 + q3 + r3;
  }
  out[threadIdx.x] += x;
}

template <typename uint>
class FastDivPerf : public ::testing::Test {};

using FastDivTypes = ::testing::Types<uint32_t, uint64_t>;
TYPED_TEST_SUITE(FastDivPerf, FastDivTypes);

TEST(FastDiv_Host, div_lohi) {
  detail::lohi<uint64_t> num;
  uint64_t den, ref, q;
  num = { 0, 1 };
  den = 2;
  q = detail::div_lohi(num.lo, num.hi, den);
  ref = 1_u64 << 63;
  EXPECT_EQ(q, ref);

  num = { 0xCAFEBABEFACEFEED_u64, 0x600DF00D_u64 };
  den = 0x600DF00E_u64;  // this divisor is only slightly larger than the high word which
                       // makes the division more prone to errors, should there be any
  // reference is calculated off-line as ((unsigned __int128)num.hi << 64 | num.lo) / den;
  ref = 0xFFFFFFFF72BBC9CE_u64;
  q = detail::div_lohi(num.lo, num.hi, den);
  EXPECT_EQ(q, ref);
}

DEVICE_TEST(FastDiv_Dev, div_lohi, 1, 1) {
  detail::lohi<uint64_t> num;
  uint64_t den, ref, q;
  num = { 0, 1 };
  den = 2;
  q = detail::div_lohi(num.lo, num.hi, den);
  ref = 1_u64 << 63;
  DEV_EXPECT_EQ(q, ref);

  num = { 0xCAFEBABEFACEFEED_u64, 0x600DF00D_u64 };
  den = 0x600DF00E_u64;  // this divisor is only slightly larger than the high word which
                       // makes the division more prone to errors, should there be any
  // reference is calculated off-line as ((unsigned __int128)num.hi << 64 | num.lo) / den;
  ref = 0xFFFFFFFF72BBC9CE_u64;
  q = detail::div_lohi(num.lo, num.hi, den);
  DEV_EXPECT_EQ(q, ref);
}

TYPED_TEST(FastDivPerf, Perf) {
  using T = TypeParam;
  CUDAEvent start, end;
  start = CUDAEvent::CreateWithFlags(0);
  end = CUDAEvent::CreateWithFlags(0);
  int64_t N = 1000000;

  std::mt19937_64 rng;
  std::uniform_int_distribution<T> dist(0, 1<<24);

  DeviceBuffer<T> m;
  m.resize(2048);

  const int divs_per_thread = 18;

  T d1 = std::max(dist(rng), T(1));
  T d2 = std::max(dist(rng), T(1));
  T d3 = std::max(dist(rng), T(1));

  FastDivMod<T><<<1000, 1024>>>(m, d1, d2, d3);
  CUDA_CALL(cudaEventRecord(start, 0));
  FastDivMod<T><<<N, 1024>>>(m, d1, d2, d3);
  CUDA_CALL(cudaEventRecord(end, 0));
  CUDA_CALL(cudaDeviceSynchronize());
  float t_fast = 0;
  CUDA_CALL(cudaEventElapsedTime(&t_fast, start, end));

  NormalDivMod<<<1000, 1024>>>(m + 1024, d1, d2, d3);
  CUDA_CALL(cudaEventRecord(start, 0));
  NormalDivMod<<<N, 1024>>>(m + 1024, d1, d2, d3);
  CUDA_CALL(cudaEventRecord(end, 0));
  CUDA_CALL(cudaDeviceSynchronize());
  float t_norm = 0;
  CUDA_CALL(cudaEventElapsedTime(&t_norm, start, end));

  t_norm *= 1e+6;
  t_fast *= 1e+6;

  std::cerr << "Normal division: " << (N * 1024 * divs_per_thread / t_norm) << " div/ns\n";
  std::cerr << "Fast division:   " << (N * 1024 * divs_per_thread / t_fast) << " div/ns\n";
}

}  // namespace dali
