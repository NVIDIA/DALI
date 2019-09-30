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
#include "dali/core/byte_io.h"

namespace dali {

namespace test {

TEST(byte_io, read_value_unsigned_int) {
  const uint8_t data[] = {0x04, 0xd2};  // dec 1234 = hex 0x04D2
  const uint8_t data_le[] = {0xd2, 0x04};
  // 1-byte, doesn't matter if it is BE/LE
  EXPECT_EQ(4, ReadValueBE<uint8_t>(data));
  EXPECT_EQ(4, ReadValueLE<uint8_t>(data));
  EXPECT_EQ(1234, ReadValueBE<uint16_t>(data));
  EXPECT_EQ(1234, ReadValueLE<uint16_t>(data_le));

  // Using nbytes < sizeof(T)
  EXPECT_EQ(1234, (ReadValueBE<uint32_t, 2>(data)));
  EXPECT_EQ(1234, (ReadValueLE<uint32_t, 2>(data_le)));
}

TEST(byte_io, read_value_signed_int) {
  const uint8_t plus_data[] = {0x00, 0x00, 0x04, 0xd2};
  const uint8_t plus_data_le[] = {0xd2, 0x04, 0x00, 0x00};
  EXPECT_EQ(1234, ReadValueBE<int32_t>(plus_data));
  EXPECT_EQ(1234, ReadValueLE<int32_t>(plus_data_le));

  // dec -1234 = 0xfffffb2e
  const uint8_t minus_data[] = {0xff, 0xff, 0xfb, 0x2e};
  const uint8_t minus_data_le[] = {0x2e, 0xfb, 0xff, 0xff};
  EXPECT_EQ(-1234, ReadValueBE<int32_t>(minus_data));
  EXPECT_EQ(-1234, ReadValueLE<int32_t>(minus_data_le));

  // nbytes < sizeo(T)
  EXPECT_EQ(1234, (ReadValueBE<int32_t, 3>(plus_data+1)));
  EXPECT_EQ(1234, (ReadValueLE<int32_t, 3>(plus_data_le)));
  EXPECT_EQ(-1234, (ReadValueBE<int32_t, 3>(minus_data+1)));
  EXPECT_EQ(-1234, (ReadValueLE<int32_t, 3>(minus_data_le)));
}

TEST(byte_io, read_value_float) {
  const uint8_t data[] = {0x3f, 0x80, 0x00, 0x00};
  const uint8_t data_le[] = {0x00, 0x00, 0x80, 0x3f};
  EXPECT_EQ(1.0f, ReadValueBE<float>(data));
  EXPECT_EQ(1.0f, ReadValueLE<float>(data_le));
}

}  // namespace test

}  // namespace dali
