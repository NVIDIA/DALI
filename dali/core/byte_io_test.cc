// Copyright (c) 2019, 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/byte_io.h"
#include "dali/core/stream.h"

namespace dali {

namespace test {

TEST(byte_io, read_value_unsigned_int) {
  const uint8_t data_be[] = {0x04, 0xd2};  // dec 1234 = hex 0x04D2
  const uint8_t data_le[] = {0xd2, 0x04};
  // 1-byte, doesn't matter if it is BE/LE
  EXPECT_EQ(4, ReadValueBE<uint8_t>(data_be));
  EXPECT_EQ(4, ReadValueLE<uint8_t>(data_be));
  EXPECT_EQ(1234, ReadValueBE<uint16_t>(data_be));
  EXPECT_EQ(1234, ReadValueLE<uint16_t>(data_le));

  // Using nbytes < sizeof(T)
  EXPECT_EQ(1234, (ReadValueBE<uint32_t, 2>(data_be)));
  EXPECT_EQ(1234, (ReadValueLE<uint32_t, 2>(data_le)));
}

TEST(byte_io, read_value_signed_int) {
  const uint8_t plus_data_be[] = {0x00, 0x00, 0x04, 0xd2};
  const uint8_t plus_data_le[] = {0xd2, 0x04, 0x00, 0x00};
  EXPECT_EQ(1234, ReadValueBE<int32_t>(plus_data_be));
  EXPECT_EQ(1234, ReadValueLE<int32_t>(plus_data_le));

  // dec -1234 = 0xfffffb2e
  const uint8_t minus_data_be[] = {0xff, 0xff, 0xfb, 0x2e};
  const uint8_t minus_data_le[] = {0x2e, 0xfb, 0xff, 0xff};
  EXPECT_EQ(-1234, ReadValueBE<int32_t>(minus_data_be));
  EXPECT_EQ(-1234, ReadValueLE<int32_t>(minus_data_le));

  // nbytes < sizeof(T)
  EXPECT_EQ(1234, (ReadValueBE<int32_t, 3>(plus_data_be+1)));
  EXPECT_EQ(1234, (ReadValueLE<int32_t, 3>(plus_data_le)));
  EXPECT_EQ(-1234, (ReadValueBE<int32_t, 3>(minus_data_be+1)));
  EXPECT_EQ(-1234, (ReadValueLE<int32_t, 3>(minus_data_le)));
}

TEST(byte_io, read_value_float) {
  const uint8_t data_be[] = {0x3f, 0x80, 0x00, 0x00};
  const uint8_t data_le[] = {0x00, 0x00, 0x80, 0x3f};
  EXPECT_EQ(1.0f, ReadValueBE<float>(data_be));
  EXPECT_EQ(1.0f, ReadValueLE<float>(data_le));
}

TEST(byte_io, read_value_unsigned_int_input_stream) {
  const uint8_t data_be[] = {0x04, 0xd2, 0x1e, 0x61};  // dec 7777 = hex 0x1E61
  const uint8_t data_le[] = {0xd2, 0x04, 0x61, 0x1e};
  MemInputStream mis_be(data_be, sizeof(data_be));
  MemInputStream mis_le(data_le, sizeof(data_le));

  EXPECT_EQ(1234, (ReadValueBE<uint32_t, 2>(mis_be)));
  EXPECT_EQ(1234, (ReadValueLE<uint32_t, 2>(mis_le)));
  EXPECT_EQ(7777, ReadValueBE<uint16_t>(mis_be));
  EXPECT_EQ(7777, ReadValueLE<uint16_t>(mis_le));
}

TEST(byte_io, read_value_float_input_stream) {
  const uint8_t data_be[] = {0x3f, 0x80, 0x00, 0x00};
  const uint8_t data_le[] = {0x00, 0x00, 0x80, 0x3f};
  MemInputStream mis_be(data_be, sizeof(data_be));
  MemInputStream mis_le(data_le, sizeof(data_le));

  EXPECT_EQ(1.0f, ReadValueBE<float>(mis_be));
  EXPECT_EQ(1.0f, ReadValueLE<float>(mis_le));
}

TEST(byte_io, read_value_enum_overload) {
  const uint8_t data_be[] = {0x00, 0x00, 0x00, 0x01};
  const uint8_t data_le[] = {0x01, 0x00, 0x00, 0x00};
  const uint8_t data_le_negative[4] = {0xD6, 0xFF, 0xFF, 0xFF };  // -42
  enum TestEnum {
    VALUE_ZERO = 0,
    VALUE_ONE = 1,
    VALUE_NEGATIVE = -42,
  };
  EXPECT_EQ(VALUE_ONE, ReadValueBE<TestEnum>(data_be));
  EXPECT_EQ(VALUE_ONE, ReadValueLE<TestEnum>(data_le));
  EXPECT_EQ(VALUE_NEGATIVE, (ReadValueLE<TestEnum, 3>(data_le_negative)));
}

TEST(byte_io, read_value_enum_overload_uint16) {
  const uint8_t data_be[] = {0x00, 0x01};
  const uint8_t data_le[] = {0x01, 0x00};
  enum TestEnum : uint16_t {
    VALUE_ZERO = 0,
    VALUE_ONE = 1,
  };
  EXPECT_EQ(VALUE_ONE, ReadValueBE<TestEnum>(data_be));
  EXPECT_EQ(VALUE_ONE, ReadValueLE<TestEnum>(data_le));
}

}  // namespace test

}  // namespace dali
