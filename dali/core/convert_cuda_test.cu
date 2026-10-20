// Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/convert.h"
#include "dali/core/int_literals.h"
#include "dali/core/math_util.h"
#include "dali/core/convert_test_static.h"
#include "dali/test/device_test.h"

namespace dali {

DEVICE_TEST(ConvertSatCUDA_Dev, float2int, 110, 1024) {
  float f = ldexpf(threadIdx.x-512.0f, blockIdx.x-10);
  float integral;
  float fract = modff(f, &integral);
  if (fract == 0.5f || fract == -0.5f)
    return;
  double rounded = roundf(f);
  int64_t clamped = clamp<double>(rounded, -128, 127);
  DEV_EXPECT_EQ(ConvertSat<int8_t>(f), clamped);
  clamped = clamp<double>(rounded, 0, 255);
  DEV_EXPECT_EQ(ConvertSat<uint8_t>(f), clamped);
  clamped = clamp<double>(rounded, -0x8000, 0x7fff);
  DEV_EXPECT_EQ(ConvertSat<int16_t>(f), clamped);
  clamped = clamp<double>(rounded, 0, 0xffff);
  DEV_EXPECT_EQ(ConvertSat<uint16_t>(f), clamped);
  clamped = clamp<double>(rounded, int32_t(~0x7fffffff), 0x7fffffff);
  DEV_EXPECT_EQ(ConvertSat<int32_t>(f), clamped);
  clamped = clamp<double>(rounded, 0, 0xffffffffu);
  DEV_EXPECT_EQ(ConvertSat<uint32_t>(f), clamped);
}

DEVICE_TEST(ConvertSatCUDA_Dev, fp2int64, 1, 1) {
  DEV_EXPECT_EQ(ConvertSat<int64_t>(123456789123.0), 123456789123ll);
  DEV_EXPECT_EQ(ConvertSat<int64_t>(-123456789123.0), -123456789123ll);
  DEV_EXPECT_EQ(ConvertSat<int64_t>(2e+20f), 0x7fffffffffffffffll);
  DEV_EXPECT_EQ(ConvertSat<int64_t>(-2e+20f), ~0x7fffffffffffffffll);
  DEV_EXPECT_EQ(ConvertSat<uint64_t>(2e+20f), 0xffffffffffffffffull);
  DEV_EXPECT_EQ(ConvertSat<uint64_t>(-1.0f), 0u);
}

DEVICE_TEST(ConvertNormCUDA_Dev, float2int, 1, 1) {
  DEV_EXPECT_EQ(ConvertNorm<uint8_t>(0.0f), 0);
  DEV_EXPECT_EQ(ConvertNorm<uint8_t>(0.499f), 127);
  DEV_EXPECT_EQ(ConvertNorm<uint8_t>(1.0f), 255);
  DEV_EXPECT_EQ(ConvertNorm<int8_t>(1.0f), 127);
  DEV_EXPECT_EQ(ConvertNorm<int8_t>(0.499f), 63);
  DEV_EXPECT_EQ(ConvertNorm<int8_t>(-1.0f), -127);

  DEV_EXPECT_EQ(ConvertNorm<uint16_t>(0.0f), 0);
  DEV_EXPECT_EQ(ConvertNorm<uint16_t>(1.0f), 0xffff);
  DEV_EXPECT_EQ(ConvertNorm<int16_t>(1.0f), 0x7fff);
  DEV_EXPECT_EQ(ConvertNorm<int16_t>(-1.0f), -0x7fff);
  DEV_EXPECT_EQ(ConvertNorm<uint16_t>(0.0f), 0);

  // float doesn't have appropriate precision, so we can only expect 23 MSBs to be valid.
  DEV_EXPECT_GE(ConvertNorm<uint64_t>(1.0f),  0xffffff0000000000);
  DEV_EXPECT_GE(ConvertNorm<int64_t>(1.0f),   0x7fffff8000000000);
  DEV_EXPECT_LE(ConvertNorm<int64_t>(-1.0f), -0x7fffff8000000000);
}

// max_value<Out>() of these types is not representable in float16, so the scaling must be done
// in float. The values are chosen so that none of the results is an exact tie.
DEVICE_TEST(ConvertNorm_Dev, float16_2int, 1, 1) {
  DEV_EXPECT_EQ(ConvertNorm<uint16_t>(float16(0.25f)), 16384);
  DEV_EXPECT_EQ(ConvertNorm<int32_t>(float16(0.5f)), 0x40000000);
  DEV_EXPECT_EQ(ConvertNorm<int16_t>(float16(1.0f)), 0x7fff);
  DEV_EXPECT_EQ(ConvertNorm<int16_t>(float16(-1.0f)), -0x7fff);
  DEV_EXPECT_EQ(ConvertNorm<int64_t>(float16(0.5f)), 0x4000000000000000);
  DEV_EXPECT_EQ(ConvertSatNorm<uint8_t>(float16(0.25f)), 64);
  DEV_EXPECT_EQ(ConvertSatNorm<uint16_t>(float16(0.0f)), 0);
  DEV_EXPECT_EQ(ConvertSatNorm<uint16_t>(float16(0.25f)), 16384);
  DEV_EXPECT_EQ(ConvertSatNorm<uint16_t>(float16(1.0f)), 0xffff);
  DEV_EXPECT_EQ(ConvertSatNorm<int32_t>(float16(0.0f)), 0);
  DEV_EXPECT_EQ(ConvertSatNorm<int32_t>(float16(0.5f)), 0x40000000);
  DEV_EXPECT_EQ(ConvertSatNorm<int32_t>(float16(2.0f)), 0x7fffffff);
  DEV_EXPECT_EQ(ConvertSatNorm<uint32_t>(float16(0.5f)), 0x80000000u);
  DEV_EXPECT_EQ(ConvertSatNorm<int16_t>(float16(-2.0f)), -0x8000);
}

DEVICE_TEST(ConvertSatNorm_Dev, float2int, 1, 1) {
  DEV_EXPECT_EQ(ConvertSatNorm<uint8_t>(2.0f), 255);
  DEV_EXPECT_EQ(ConvertSatNorm<uint8_t>(0.499f), 127);
  DEV_EXPECT_EQ(ConvertSatNorm<uint8_t>(-2.0f), 0);
  DEV_EXPECT_EQ(ConvertSatNorm<int8_t>(2.0f), 127);
  DEV_EXPECT_EQ(ConvertSatNorm<int8_t>(0.499f), 63);
  DEV_EXPECT_EQ(ConvertSatNorm<int8_t>(-2.0f), -128);
  DEV_EXPECT_EQ(ConvertSatNorm<uint8_t>(0.4f/255), 0);
  DEV_EXPECT_EQ(ConvertSatNorm<uint8_t>(0.6f/255), 1);

  DEV_EXPECT_EQ(ConvertSatNorm<int16_t>(2.0f), 0x7fff);
  DEV_EXPECT_EQ(ConvertSatNorm<int16_t>(-2.0f), -0x8000);

  DEV_EXPECT_GE(ConvertSatNorm<int64_t>(2.0f),  0x7fffff8000000000);
  DEV_EXPECT_LE(ConvertSatNorm<int64_t>(-2.0f), -0x7fffff8000000000);
  DEV_EXPECT_EQ(ConvertSatNorm<uint64_t>(-2.0f), 0u);
}

template <typename FP>
__device__ FP infinity(int sign) {
  static_assert(sizeof(FP) == 2 || sizeof(FP) == 4 || sizeof(FP) == 8);
  if constexpr (sizeof(FP) == 2) {
    uint16_t x = 0x7c00u;
    if (sign < 0)
      x |= 0x8000;
    return *reinterpret_cast<FP *>(&x);
  } else if constexpr (sizeof(FP) == 8) {  // NOLINT - linter bug with else if constexpr
    uint64_t x = 0x7ff0000000000000_u64;
    if (sign < 0)
      x |= 0x8000000000000000_u64;
    return *reinterpret_cast<FP *>(&x);
  } else {  /* 32-bit */ // NOLINT - linter bug with else if constexpr
    uint32_t x = 0x7f800000u;
    if (sign < 0)
      x |= 0x80000000u;
    return *reinterpret_cast<FP *>(&x);
  }
}

template <typename FP>
__device__ FP denorm(int sign) {
  if constexpr (sizeof(FP) == 2) {
    uint16_t x = 0x1;
    if (sign < 0)
      x |= 0x8000;
    return *reinterpret_cast<FP *>(&x);
  } else if constexpr (sizeof(FP) == 8) {  // NOLINT - linter bug with else if constexpr
    uint64_t x = 1;
    if (sign < 0)
      x |= 0x8000000000000000_u64;
    return *reinterpret_cast<FP *>(&x);
  } else {  /* 32-bit */ // NOLINT - linter bug with else if constexpr
    uint32_t x = 1;
    if (sign < 0)
      x |= 0x80000000u;
    return *reinterpret_cast<FP *>(&x);
  }
}

template <typename FP>
__device__ void CheckFP2Bool(testing::TestStatusBlock &test_status) {
  // Values in (-0.5, 0.5) would be rounded to 0 if the conversion went through rounding.
  const double nonzero[] = { 0.1, -0.1, 0.4, -0.4, 0.5, -0.5, 0.9, 1.0, -1.0, 1e-3, 1000.0 };
  for (double v : nonzero) {
    FP x = v;
    DEV_EXPECT_TRUE(Convert<bool>(x));
    DEV_EXPECT_TRUE(ConvertSat<bool>(x));
    DEV_EXPECT_TRUE(ConvertNorm<bool>(x));
    DEV_EXPECT_TRUE(ConvertSatNorm<bool>(x));
  }
  const FP zeros[] = { FP(0.0), FP(-0.0) };
  for (FP x : zeros) {
    DEV_EXPECT_FALSE(Convert<bool>(x));
    DEV_EXPECT_FALSE(ConvertSat<bool>(x));
    DEV_EXPECT_FALSE(ConvertNorm<bool>(x));
    DEV_EXPECT_FALSE(ConvertSatNorm<bool>(x));
  }
  FP nan = NAN;
  DEV_ASSERT_TRUE(isnan(static_cast<float>(nan)));
  DEV_EXPECT_TRUE(Convert<bool>(nan));
  DEV_EXPECT_TRUE(ConvertSat<bool>(nan));
  DEV_EXPECT_TRUE(ConvertNorm<bool>(nan));
  DEV_EXPECT_TRUE(ConvertSatNorm<bool>(nan));

  for (auto inf : { infinity<FP>(1), infinity<FP>(-1) }) {
    DEV_ASSERT_TRUE(isinf(static_cast<float>(inf)));
    DEV_EXPECT_TRUE(Convert<bool>(inf));
    DEV_EXPECT_TRUE(ConvertSat<bool>(inf));
    DEV_EXPECT_TRUE(ConvertNorm<bool>(inf));
    DEV_EXPECT_TRUE(ConvertSatNorm<bool>(inf));
  }

  for (auto den : { denorm<FP>(1), denorm<FP>(-1) }) {
    DEV_EXPECT_TRUE(Convert<bool>(den));
    DEV_EXPECT_TRUE(ConvertSat<bool>(den));
    DEV_EXPECT_TRUE(ConvertNorm<bool>(den));
    DEV_EXPECT_TRUE(ConvertSatNorm<bool>(den));
  }
}

DEVICE_TEST(ConvertCUDA_Dev, fp2bool, 1, 1) {
  CheckFP2Bool<float>(test_status);
  CheckFP2Bool<double>(test_status);
  CheckFP2Bool<float16>(test_status);
}

DEVICE_TEST(ConvertSatNorm_Dev, int2int, 1, 1) {
  DEV_EXPECT_EQ((ConvertSatNorm<uint8_t, int8_t>(-1)), 0);
  DEV_EXPECT_EQ((ConvertSatNorm<uint8_t, int16_t>(-0x8000)), 0);
  DEV_EXPECT_EQ((ConvertSatNorm<uint8_t, int16_t>(0x7fff)), 255);
  DEV_EXPECT_EQ((ConvertSatNorm<uint16_t, int32_t>(0x7fffffff)), 0xffff);
  DEV_EXPECT_EQ((ConvertSatNorm<uint16_t, uint8_t>(255)), 0xffff);
  DEV_EXPECT_EQ((ConvertSatNorm<int8_t, int16_t>(0x7fff)), 127);
  DEV_EXPECT_EQ((ConvertSatNorm<int16_t, int8_t>(-127)), -0x7fff);
  DEV_EXPECT_EQ((ConvertSatNorm<int16_t, int8_t>(-128)), -0x8000);
  DEV_EXPECT_EQ((ConvertSatNorm<int8_t, int16_t>(-0x8000)), -127);
  DEV_EXPECT_EQ((ConvertSatNorm<int8_t, int8_t>(-128)), -128);
}

TEST(ConvertNorm_CUDA_Host, int2float) {
  EXPECT_EQ((ConvertNorm<float, uint8_t>(255)), 1.0f);
  EXPECT_NEAR((ConvertNorm<float, uint8_t>(127)), 1.0f*127/255, 1e-7f);
  EXPECT_EQ((ConvertNorm<float, int8_t>(127)), 1.0f);
  EXPECT_NEAR((ConvertNorm<float, int8_t>(64)), 1.0f*64/127, 1e-7f);
}

}  // namespace dali
