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
#include "dali/core/stream.h"

namespace dali {

TEST(MemInputStreamTest, ReadAndSeek) {
  char mem[] = "test123";

  MemInputStream mis(mem, 7);
  char out[8] = {0};
  EXPECT_EQ(mis.Read(out, 8), 7);
  EXPECT_STREQ(out, "test123");
  std::memset(out, 0, sizeof(out));
  mis.SeekRead(1, SEEK_SET);
  EXPECT_EQ(mis.TellRead(), 1);
  mis.SeekRead(1, SEEK_CUR);
  EXPECT_EQ(mis.TellRead(), 2);
  EXPECT_EQ(mis.Read(out, 3), 3);
  EXPECT_STREQ(out, "st1");

  std::memset(out, 0, sizeof(out));
  mis.SeekRead(-2, SEEK_END);
  EXPECT_EQ(mis.TellRead(), 5);
  EXPECT_EQ(mis.Read(out, 3), 2);
  EXPECT_STREQ(out, "23");
}

TEST(MemInputStreamTest, ReadBytes) {
  char mem[] = "test123";

  MemInputStream mis(mem, 7);
  char out[8] = {0};
  mis.ReadAll(out, 5);
  EXPECT_STREQ(out, "test1");
}


TEST(MemInputStreamTest, Errors) {
  char mem[] = "test123";

  MemInputStream mis(mem, 7);
  char out[8] = {0};
  mis.ReadAll(out, 5);
  EXPECT_THROW(mis.ReadBytes(out, 8), EndOfStream);
  mis.SeekRead(0);
  EXPECT_THROW(mis.SeekRead(-1, SEEK_SET), std::out_of_range);
}

TEST(MemInputStreamTest, ReadTyped) {
  char mem[32];
  int ofs = 0;
  mem[ofs++] = 'x';
  ofs += sizeof(float);
  *reinterpret_cast<int *>(mem + ofs) = 0x12345678;
  ofs += sizeof(int);
  *reinterpret_cast<uint16_t *>(mem + ofs) = 0x600Du;
  ofs += sizeof(uint16_t);
  *reinterpret_cast<uint16_t *>(mem + ofs) = 0xF00Du;
  ofs += sizeof(uint16_t);

  MemInputStream mis(mem, ofs);
  EXPECT_EQ((mis.ReadOne<char>()), 'x');
  mis.Skip<float>();
  EXPECT_EQ((mis.ReadOne<int>()), 0x12345678);
  uint16_t s[2];
  mis.ReadAll(s, 2);
  EXPECT_EQ(s[0], 0x600Du);
  EXPECT_EQ(s[1], 0xF00Du);
}

}  // namespace dali
