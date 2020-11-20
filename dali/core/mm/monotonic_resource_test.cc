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

#include <gtest/gtest.h>
#include <dali/core/mm/monotonic_resource.h>

namespace dali {
namespace mm {

TEST(MMTest, MonotonicBufferResource) {
  static char buf alignas(1024)[1024];
  monotonic_buffer_resource mr(buf, sizeof(buf));
  char *a0 = static_cast<char*>(mr.allocate(1, 1));
  EXPECT_EQ(a0, buf);
  char *a1 = static_cast<char*>(mr.allocate(1, 8));
  EXPECT_EQ(a1, buf+8);
  char *a2 = static_cast<char*>(mr.allocate(7, 1));
  EXPECT_EQ(a2, buf+9);
  char *a3 = static_cast<char*>(mr.allocate(7, 1));
  EXPECT_EQ(a3, a2+7);
  char *a4 = static_cast<char*>(mr.allocate(255, 256));
  EXPECT_EQ(a4, buf+256);  // consumed 511 bytes

  // exceed by 1 byte
  EXPECT_THROW(mr.allocate(513, 2), std::bad_alloc);
  // fill up
  EXPECT_NO_THROW(mr.allocate(513, 1));
  // exceed by 1 byte
  EXPECT_THROW(mr.allocate(1, 1), std::bad_alloc);
}

TEST(MMTest, MonotonicHostResource) {
  static char buf[1<<16];
  monotonic_buffer_resource mr(buf, sizeof(buf));

}

}  // namespace mm
}  // namespace dali
