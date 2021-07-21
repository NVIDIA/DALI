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
#include "dali/core/mm/monotonic_resource.h"
#include "dali/core/mm/malloc_resource.h"
#include "dali/core/mm/mm_test_utils.h"

namespace dali {
namespace mm {
namespace test {

TEST(MMTest, MonotonicBufferResource) {
  static char buf alignas(1024)[1024];
  monotonic_buffer_resource<memory_kind::host> mr(buf, sizeof(buf));
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
  test_host_resource upstream;
  {
    monotonic_host_resource mr(&upstream);
    void *m1 = mr.allocate(100);
    ASSERT_NE(m1, nullptr);
    memset(m1, 0xff, 100);
    void *m2 = mr.allocate(256, 32);
    ASSERT_NE(m2, nullptr);
    memset(m2, 0xfe, 256);
    EXPECT_GE(static_cast<char *>(m2), static_cast<char *>(m1) + 100);
    EXPECT_TRUE(detail::is_aligned(m2, 32));
    EXPECT_LE(static_cast<char *>(m2), static_cast<char *>(m1) + align_up(100, 32));
    void *m3 = mr.allocate(1024);
    ASSERT_NE(m3, nullptr);
    memset(m3, 0xfd, 1024);
    void *m4 = mr.allocate(64000);
    ASSERT_NE(m4, nullptr);
    memset(m4, 0xfc, 64000);
    upstream.simulate_out_of_memory(true);
    EXPECT_THROW(mr.allocate(64000), std::bad_alloc);
  }
  upstream.check_leaks();
}

TEST(MMTest, MonotonicDeviceResource) {
  test_device_resource upstream;
  {
    monotonic_device_resource<> mr(&upstream);
    void *m1 = mr.allocate(100);
    ASSERT_NE(m1, nullptr);
    CUDA_CALL(cudaMemset(m1, 0xff, 100));
    void *m2 = mr.allocate(256, 32);
    ASSERT_NE(m2, nullptr);
    CUDA_CALL(cudaMemset(m2, 0xfe, 256));
    EXPECT_GE(static_cast<char *>(m2), static_cast<char *>(m1) + 100);
    EXPECT_TRUE(detail::is_aligned(m2, 32));
    EXPECT_LE(static_cast<char *>(m2), static_cast<char *>(m1) + align_up(100, 32));
    void *m3 = mr.allocate(1024);
    ASSERT_NE(m3, nullptr);
    CUDA_CALL(cudaMemset(m3, 0xfd, 1024));
    void *m4 = mr.allocate(64000);
    ASSERT_NE(m4, nullptr);
    CUDA_CALL(cudaMemset(m4, 0xfc, 64000));
  }
  CUDA_CALL(cudaDeviceSynchronize());
  upstream.check_leaks();
}

}  // namespace test
}  // namespace mm
}  // namespace dali
