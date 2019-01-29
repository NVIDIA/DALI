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

#ifndef DALI_KERNELS_COMMON_CONVERT_TEST_STATIC_H_
#define DALI_KERNELS_COMMON_CONVERT_TEST_STATIC_H_

#include "dali/kernels/common/convert.h"

namespace dali {
namespace kernels {

static_assert(clamp<uint8_t>(0) == 0, "Unexpected clamp result");
static_assert(clamp<uint8_t>(255) == 255, "Unexpected clamp result");
static_assert(clamp<uint8_t>(100) == 100, "Unexpected clamp result");
static_assert(clamp<uint8_t>(100.3) == 100, "Unexpected clamp result");
static_assert(clamp<uint8_t>(256) == 255, "Unexpected clamp result");
static_assert(clamp<uint8_t>(-4) == 0, "Unexpected clamp result");
static_assert(clamp<uint8_t>(-4.0f) == 0, "Unexpected clamp result");
static_assert(clamp<uint8_t>(1e+20f) == 255, "Unexpected clamp result");
static_assert(clamp<uint8_t>(-1e+20f) == 0, "Unexpected clamp result");
static_assert(clamp<uint8_t>(1e+200) == 255, "Unexpected clamp result");
static_assert(clamp<uint8_t>(-1e+200) == 0, "Unexpected clamp result");

static_assert(clamp<int8_t>(-4) == -4, "Unexpected clamp result");
static_assert(clamp<int8_t>(-4.2) == -4, "Unexpected clamp result");
static_assert(clamp<int8_t>(4.2) == 4, "Unexpected clamp result");
static_assert(clamp<int8_t>(127) == 127, "Unexpected clamp result");
static_assert(clamp<int8_t>(128) == 127, "Unexpected clamp result");
static_assert(clamp<int8_t>(256) == 127, "Unexpected clamp result");
static_assert(clamp<int8_t>(-128) == -128, "Unexpected clamp result");
static_assert(clamp<int8_t>(-256) == -128, "Unexpected clamp result");
static_assert(clamp<int8_t>(1e+20f) == 127, "Unexpected clamp result");
static_assert(clamp<int8_t>(-1e+20f) == -128, "Unexpected clamp result");
static_assert(clamp<int8_t>(1e+200) == 127, "Unexpected clamp result");
static_assert(clamp<int8_t>(-1e+200) == -128, "Unexpected clamp result");


static_assert(clamp<uint16_t>(0) == 0, "Unexpected clamp result");
static_assert(clamp<uint16_t>(0xffff) == 0xffff, "Unexpected clamp result");
static_assert(clamp<uint16_t>(100) == 100, "Unexpected clamp result");
static_assert(clamp<uint16_t>(100.3) == 100, "Unexpected clamp result");
static_assert(clamp<uint16_t>(0x10000) == 0xffff, "Unexpected clamp result");
static_assert(clamp<uint16_t>(-4) == 0, "Unexpected clamp result");
static_assert(clamp<uint16_t>(-4.0f) == 0, "Unexpected clamp result");
static_assert(clamp<uint16_t>(1e+20f) == 0xffff, "Unexpected clamp result");
static_assert(clamp<uint16_t>(-1e+20f) == 0, "Unexpected clamp result");
static_assert(clamp<uint16_t>(1e+200) == 0xffff, "Unexpected clamp result");
static_assert(clamp<uint16_t>(-1e+200) == 0, "Unexpected clamp result");


static_assert(clamp<int16_t>(-4) == -4, "Unexpected clamp result");
static_assert(clamp<int16_t>(-4.2) == -4, "Unexpected clamp result");
static_assert(clamp<int16_t>(4.2) == 4, "Unexpected clamp result");
static_assert(clamp<int16_t>(0x7fff) == 0x7fff, "Unexpected clamp result");
static_assert(clamp<int16_t>(0x8000) == 0x7fff, "Unexpected clamp result");
static_assert(clamp<int16_t>(0x10000) == 0x7fff, "Unexpected clamp result");
static_assert(clamp<int16_t>(-0x8000) == -0x8000, "Unexpected clamp result");
static_assert(clamp<int16_t>(-0x10000) == -0x8000, "Unexpected clamp result");
static_assert(clamp<int16_t>(1e+20f) == 0x7fff, "Unexpected clamp result");
static_assert(clamp<int16_t>(-1e+20f) == -0x8000, "Unexpected clamp result");
static_assert(clamp<int16_t>(1e+200) == 0x7fff, "Unexpected clamp result");
static_assert(clamp<int16_t>(-1e+200) == -0x8000, "Unexpected clamp result");

static_assert(needs_clamp<float, uint32_t>::value, "Invalid clamp requirement");

static_assert(sizeof(int) == 4, "Expected 32-bit integer");

static_assert(clamp<uint32_t>(0) == 0, "Unexpected clamp result");
static_assert(clamp<uint32_t>(0xffffffffLL) == 0xffffffffLL, "Unexpected clamp result");
static_assert(clamp<uint32_t>(100) == 100, "Unexpected clamp result");
static_assert(clamp<uint32_t>(100.3) == 100, "Unexpected clamp result");
static_assert(clamp<uint32_t>(0x100000000LL) == 0xffffffffLL, "Unexpected clamp result");
static_assert(clamp<uint32_t>(-4) == 0, "Unexpected clamp result");
static_assert(clamp<uint32_t>(-4.0f) == 0, "Unexpected clamp result");
static_assert(clamp<uint32_t>(1e+20f) == 0xffffffffu, "Unexpected clamp result");
static_assert(clamp<uint32_t>(-1.0e+20f) == 0, "Unexpected clamp result");
static_assert(clamp<uint32_t>(1e+200) == 0xffffffffu, "Unexpected clamp result");
static_assert(clamp<uint32_t>(-1.0e+200) == 0, "Unexpected clamp result");

static_assert(clamp<int32_t>(-4) == -4, "Unexpected clamp result");
static_assert(clamp<int32_t>(-4LL) == -4, "Unexpected clamp result");
static_assert(clamp<int32_t>(-4.2) == -4, "Unexpected clamp result");
static_assert(clamp<int32_t>(4.2) == 4, "Unexpected clamp result");
static_assert(clamp<int32_t>(0x7fffffff) == 0x7fffffff, "Unexpected clamp result");
static_assert(clamp<int32_t>(0x80000000L) == 0x7fffffff, "Unexpected clamp result");
static_assert(clamp<int32_t>(0x100000000L) == 0x7fffffff, "Unexpected clamp result");
static_assert(clamp<int32_t>(-0x80000000LL) == -0x7fffffff-1, "Unexpected clamp result");
static_assert(clamp<int32_t>(-0x100000000LL) == -0x7fffffff-1, "Unexpected clamp result");
static_assert(clamp<int32_t>(1.0e+20f) == 0x7fffffff, "Unexpected clamp result");
static_assert(clamp<int32_t>(-1.0e+20f) == -0x80000000L, "Unexpected clamp result");
static_assert(clamp<int32_t>(1.0e+200) == 0x7fffffff, "Unexpected clamp result");
static_assert(clamp<int32_t>(-1.0e+200) == -0x80000000L, "Unexpected clamp result");

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_CONVERT_TEST_STATIC_H_
