// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include <sstream>
#include <string>
#include "dali/test/dali_test.h"
#include "dali/image/tiff.h"

namespace dali {

class TiffDecoderTest : public DALITest {
 protected:
  std::string bin = {'\x41', '\x42', '\x43', '\x44', '\x45', '\x46', '\x47', '\x48', '\x49',
                     '\x4a', '\x4b', '\x4c', '\x4d', '\x4e', '\x4f', '\x50'};
};

TEST_F(TiffDecoderTest, TiffBufferBigEndianTest) {
  TiffBuffer buf(bin);
  EXPECT_EQ(65, buf.Read<int8_t>());
  EXPECT_EQ(16961, buf.Read<int16_t>());
  EXPECT_EQ(1145258561, buf.Read<int32_t>());
  EXPECT_EQ(5208208757389214273, buf.Read<int64_t>());
  EXPECT_EQ(65, buf.Read<uint8_t>());
  EXPECT_EQ(16961, buf.Read<uint16_t>());
  EXPECT_EQ(1145258561, buf.Read<uint32_t>());
  EXPECT_EQ(5208208757389214273, buf.Read<uint64_t>());
}


TEST_F(TiffDecoderTest, TiffBufferLittleEndianTest) {
  TiffBuffer buf(bin, true);
  EXPECT_EQ(65, buf.Read<int8_t>());
  EXPECT_EQ(16706, buf.Read<int16_t>());
  EXPECT_EQ(1094861636, buf.Read<int32_t>());
  EXPECT_EQ(4702394921427289928, buf.Read<int64_t>());
  EXPECT_EQ(65, buf.Read<uint8_t>());
  EXPECT_EQ(16706, buf.Read<uint16_t>());
  EXPECT_EQ(1094861636, buf.Read<uint32_t>());
  EXPECT_EQ(4702394921427289928, buf.Read<uint64_t>());
}


TEST_F(TiffDecoderTest, TiffBufferOffsetTest) {
  TiffBuffer buf_big(bin);
  EXPECT_EQ(75, buf_big.Read<int8_t>(10));
  EXPECT_EQ(19274, buf_big.Read<int16_t>(9));
  TiffBuffer buf_little(bin, true);
  EXPECT_EQ(75, buf_little.Read<int8_t>(10));
  EXPECT_EQ(1145390663, buf_little.Read<int32_t>(3));
  EXPECT_EQ(4774735094265366601, buf_little.Read<int64_t>(1));
}

}  // namespace dali
