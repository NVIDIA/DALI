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

#include "dali/core/fast_div.h"  // NOLINT
#include <gtest/gtest.h>
#include <random>
#include <iostream>
#include "dali/test/device_test.h"
#include "dali/core/dev_buffer.h"
#include "dali/core/cuda_event.h"

namespace dali {

template <uint32_t divisor>
void TestDiv32() {
  fast_div<uint32_t> fast = divisor;
  std::cerr << "Testing x / " << divisor << std::endl;
  for (uint32_t x = 0; x < 0xffffffffu; x++) {
    ASSERT_EQ(x / divisor, x / fast) << " when dividing " << x << " / " << divisor;
  }
}

// This test is disabled because it's ridiculously slow - only run when touching fast_div
TEST(FastDiv, DISABLED_U32_Host) {
  // use compile-time constants so the compiler can optimize the reference division
  TestDiv32<0xFFFFFFFEu>();
  TestDiv32<3>();
  TestDiv32<5>();
  TestDiv32<7>();
  TestDiv32<1>();
  TestDiv32<0x800000>();
  TestDiv32<19>();
  TestDiv32<42>();
  TestDiv32<12345678>();
  TestDiv32<0x82345678>();
}

template <uint64_t divisor>
void TestDiv64() {
  fast_div<uint64_t> fast = divisor;
  std::cerr << "Testing x / " << divisor << std::endl;
  for (uint64_t xx = 0; xx < 0xffffffffu; xx++) {
    uint64_t x = ((xx >> 12) << 44) + (xx & 0xfff);
    ASSERT_EQ(x / divisor, x / fast) << " when dividing " << x << " / " << divisor;
  }
}

// This test is disabled because it's ridiculously slow - only run when touching fast_div
TEST(FastDiv, DISABLED_U64_Host) {
  // use compile-time constants so the compiler can optimize the reference division
  TestDiv64<0xFFFFFFFFFFFFFFFEuL>();
  TestDiv64<3>();
  TestDiv64<5>();
  TestDiv64<7>();
  TestDiv64<1>();
  TestDiv64<0x800000>();
  TestDiv64<0x80000000000000uL>();
  TestDiv64<19>();
  TestDiv64<42>();
  TestDiv64<12345678901234567uL>();
  TestDiv64<0x82345678DEADBEEFuL>();
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
DEVICE_TEST(FastDiv, DISABLED_U64_GPU, dim3(1<<10, 1<<10), (1<<10)) {
  uint64_t start = static_cast<uint64_t>(blockIdx.x * blockDim.x + threadIdx.x) << 44;
  uint64_t divisor = blockIdx.y + 1;
  fast_div<uint64_t> fast;
  fast.init(divisor);
  uint64_t end = start + (1<<12);
  if (end == 0)
    end--;
  for (uint64_t value = start; value < end; value++) {
    auto ref = value / divisor;
    auto result = value / fast;
    if (result != ref) {
      printf("%lld / %lld   got %lld expected %lld\n", value, divisor, result, ref);
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
  ref = 1uL << 63;
  EXPECT_EQ(q, ref);

  num = { 0xCAFEBABEFACEFEEDuL, 0x600DF00DuL };
  den = 0x600DF00EuL;  // this divisor is only slightly larger than the high word which
                       // makes the division more prone to errors, should there be any
  // reference is calculated off-line as ((unsigned __int128)num.hi << 64 | num.lo) / den;
  ref = 0xFFFFFFFF72BBC9CEuL;
  q = detail::div_lohi(num.lo, num.hi, den);
  EXPECT_EQ(q, ref);
}

DEVICE_TEST(FastDiv_Dev, div_lohi, 1, 1) {
  detail::lohi<uint64_t> num;
  uint64_t den, ref, q;
  num = { 0, 1 };
  den = 2;
  q = detail::div_lohi(num.lo, num.hi, den);
  ref = 1uL << 63;
  DEV_EXPECT_EQ(q, ref);

  num = { 0xCAFEBABEFACEFEEDuL, 0x600DF00DuL };
  den = 0x600DF00EuL;  // this divisor is only slightly larger than the high word which
                       // makes the division more prone to errors, should there be any
  // reference is calculated off-line as ((unsigned __int128)num.hi << 64 | num.lo) / den;
  ref = 0xFFFFFFFF72BBC9CEuL;
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

  T d1 = max(dist(rng), T(1));
  T d2 = max(dist(rng), T(1));
  T d3 = max(dist(rng), T(1));

  FastDivMod<T><<<1000, 1024>>>(m, d1, d2, d3);
  cudaEventRecord(start, 0);
  FastDivMod<T><<<N, 1024>>>(m, d1, d2, d3);
  cudaEventRecord(end, 0);
  CUDA_CALL(cudaDeviceSynchronize());
  float t_fast = 0;
  cudaEventElapsedTime(&t_fast, start, end);

  NormalDivMod<<<1000, 1024>>>(m + 1024, d1, d2, d3);
  cudaEventRecord(start, 0);
  NormalDivMod<<<N, 1024>>>(m + 1024, d1, d2, d3);
  cudaEventRecord(end, 0);
  CUDA_CALL(cudaDeviceSynchronize());
  float t_norm = 0;
  cudaEventElapsedTime(&t_norm, start, end);

  t_norm *= 1e+6;
  t_fast *= 1e+6;

  std::cerr << "Normal division: " << (N * 1024 * divs_per_thread / t_norm) << " div/ns\n";
  std::cerr << "Fast division:   " << (N * 1024 * divs_per_thread / t_fast) << " div/ns\n";
}

}  // namespace dali
