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

#include "tiff.h"
#include <dali/test/dali_test.h>

#include <sstream>

using namespace std;

namespace dali {

class TiffDecoderTest : public DALITest {
protected:
    std::string bin = {'\x41', '\x42', '\x43', '\x44', '\x45', '\x46', '\x47', '\x48', '\x49', '\x4a', '\x4b', '\x4c',
                       '\x4d', '\x4e', '\x4f', '\x50'};
};

TEST_F(TiffDecoderTest, TiffBufferBigEndianTest) {
    tiff_buffer buf(bin);
    EXPECT_EQ(65, buf.read<char>());
    EXPECT_EQ(16961, buf.read<short>());
    EXPECT_EQ(1145258561, buf.read<int>());
    EXPECT_EQ(5208208757389214273, buf.read<long>());
    EXPECT_EQ(65, buf.read<unsigned char>());
    EXPECT_EQ(16961, buf.read<unsigned short>());
    EXPECT_EQ(1145258561, buf.read<unsigned int>());
    EXPECT_EQ(5208208757389214273, buf.read<unsigned long>());
}


TEST_F(TiffDecoderTest, TiffBufferLittleEndianTest) {
    tiff_buffer buf(bin, true);
    EXPECT_EQ(65, buf.read<char>());
    EXPECT_EQ(16706, buf.read<short>());
    EXPECT_EQ(1094861636, buf.read<int>());
    EXPECT_EQ(4702394921427289928, buf.read<long>());
    EXPECT_EQ(65, buf.read<unsigned char>());
    EXPECT_EQ(16706, buf.read<unsigned short>());
    EXPECT_EQ(1094861636, buf.read<unsigned int>());
    EXPECT_EQ(4702394921427289928, buf.read<unsigned long>());
}


TEST_F(TiffDecoderTest, TiffBufferOffsetTest) {
    tiff_buffer buf_big(bin);
    EXPECT_EQ(75, buf_big.read<char>(10));
    EXPECT_EQ(19274, buf_big.read<short>(9));
    tiff_buffer buf_little(bin, true);
    EXPECT_EQ(75, buf_little.read<char>(10));
    EXPECT_EQ(1145390663, buf_little.read<int>(3));
    EXPECT_EQ(4774735094265366601, buf_little.read<long>(1));
}

}  // namespace dali
