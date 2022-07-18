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

#include <gtest/gtest.h>
#include <dali/core/endian_util.h>
#include <dali/core/int_literals.h>

namespace dali {

TEST(EndianUtilTest, Fundamental) {
  int32_t a = 0x12345678;
  swap_endian(a);
  EXPECT_EQ(a, 0x78563412);
  uint64_t b = 0x0123456789abcdef_u64;
  swap_endian(b);
  EXPECT_EQ(b, 0xefcdab8967452301_u64);
  char c = 'x';
  EXPECT_EQ(c, 'x');
  uint32_t f_ieee_swapped = 0x0000803f;
  float f = *(const float *)&f_ieee_swapped;
  swap_endian(f);
  EXPECT_EQ(f, 1);
}

namespace {

struct Inner {
  int16_t x, y;
};

struct Outer {
  Inner inner;
  int a[2];
  std::array<int, 2> b;
};

}  // namespace

SWAP_ENDIAN_FIELDS(Inner, x, y);

SWAP_ENDIAN_FIELDS(Outer, inner, a, b);


TEST(EndianUtilTest, Struct) {
  Outer o = { { 0xab12_i16, 0xcd34_i16 },
              { static_cast<int>(0xab01cd23), 0x01237654 },
              { static_cast<int>(0xab01cd23), 0x01237654 } };
  if (is_little_endian)
    from_big_endian(o);
  else
    from_little_endian(o);
  EXPECT_EQ(o.inner.x, 0x12ab);
  EXPECT_EQ(o.inner.y, 0x34cd);
  EXPECT_EQ(o.a[0], 0x23cd01ab);
  EXPECT_EQ(o.a[1], 0x54762301);
  EXPECT_EQ(o.b[0], 0x23cd01ab);
  EXPECT_EQ(o.b[1], 0x54762301);
}

TEST(EndianUtilTest, Pair) {
  std::pair<uint32_t, uint16_t> p = { 0x01237654, 0x1234 };
  swap_endian(p);
  EXPECT_EQ(p, (std::pair<uint32_t, uint16_t>(0x54762301, 0x3412)));
}

TEST(EndianUtilTest, Tuple) {
  std::tuple<uint32_t, uint16_t, char> p = { 0x01237654, 0x1234, 'c' };
  swap_endian(p);
  EXPECT_EQ(p, (std::tuple<uint32_t, uint16_t, char>(0x54762301, 0x3412, 'c')));
}

}  // namespace dali
