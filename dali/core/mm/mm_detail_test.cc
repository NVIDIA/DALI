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
#include "dali/core/mm/detail/align.h"
#include "dali/core/mm/mm_test_utils.h"

namespace dali {
namespace mm {
namespace test {

TEST(MMTest, AllignedAlloc) {
  static char buf alignas(4096)[(1<<16)];

  for (int alignment = 1; alignment <= 4096; alignment += alignment) {
    int min_headroom = alignment;
    for (int offset = 0; offset <= alignment; offset++) {
      size_t size = 123;
      size_t upstream_size = 0;
      char *unaligned = buf + offset;
      char *ptr = static_cast<char*>(detail::aligned_alloc([&](size_t size) {
        upstream_size = size;
        return unaligned;
      }, size, alignment));
      EXPECT_LE(ptr + size, unaligned + upstream_size) << "Too little space requested";
      int headroom = (unaligned + upstream_size) - (ptr + size);
      if (headroom < min_headroom)
        min_headroom = headroom;
      EXPECT_TRUE(detail::is_aligned(ptr, alignment)) << "Not aligned";
      detail::aligned_dealloc([&](void *mem, size_t size) {
        EXPECT_EQ(mem, unaligned) << "Dealloc got a different pointer than was returned by alloc.";
        EXPECT_EQ(size, upstream_size) << "Dealloc got a different size than was passed to alloc";
      }, ptr, size, alignment);
    }
    EXPECT_EQ(min_headroom, 0) << "aligned_alloc always requests too much memory";
  }
}

}  // namespace test
}  // namespace mm
}  // namespace dali
