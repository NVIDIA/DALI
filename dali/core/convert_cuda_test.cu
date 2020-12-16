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

#include <gtest/gtest.h>
#include "dali/core/convert.h"
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

TEST(ConvertNorm_CUDA_Host, int2float) {
  EXPECT_EQ((ConvertNorm<float, uint8_t>(255)), 1.0f);
  EXPECT_NEAR((ConvertNorm<float, uint8_t>(127)), 1.0f*127/255, 1e-7f);
  EXPECT_EQ((ConvertNorm<float, int8_t>(127)), 1.0f);
  EXPECT_NEAR((ConvertNorm<float, int8_t>(64)), 1.0f*64/127, 1e-7f);
}

}  // namespace dali
