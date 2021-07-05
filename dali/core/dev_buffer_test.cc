// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/dev_buffer.h"

namespace dali {

TEST(DeviceBuffer, SizeChanges) {
  DeviceBuffer<int> db;
  db.resize(1);
  EXPECT_EQ(db.size(), 1u);
  EXPECT_GE(db.capacity(), 1);
  db.resize(2);
  EXPECT_EQ(db.size(), 2u);
  EXPECT_GE(db.capacity(), 2u);
  db.resize(3);
  EXPECT_EQ(db.size(), 3u);
  EXPECT_GE(db.capacity(), 4u);
  auto *ptr = db.data();
  db.resize(4);
  EXPECT_EQ(db.size(), 4u);
  EXPECT_GE(db.capacity(), 4u);
  EXPECT_EQ(db.data(), ptr) << "No reallocation should have occurred";
  db.resize(5);
  EXPECT_EQ(db.size(), 5u);
  EXPECT_GE(db.capacity(), 8u);
  ptr = db.data();
  db.resize(8);
  EXPECT_EQ(db.size(), 8u);
  EXPECT_GE(db.capacity(), 8u);
  EXPECT_EQ(db.data(), ptr) << "No reallocation should have occurred";
  db.resize(3);
  EXPECT_EQ(db.size(), 3u);
  EXPECT_GE(db.capacity(), 8u);
  EXPECT_EQ(db.data(), ptr) << "No reallocation should have occurred";
  db.shrink_to_fit();
  EXPECT_GE(db.capacity(), 3u);
  EXPECT_NE(db.data(), ptr) << "Buffer contents must have been copied - no in-place reallocation";
  ptr = db.data();
  db.clear();
  EXPECT_EQ(db.size(), 0u);
  EXPECT_GE(db.capacity(), 3u);
  EXPECT_EQ(db.data(), ptr) << "No reallocation should have occurred";
  db.shrink_to_fit();
  EXPECT_EQ(db.size(), 0u);
  EXPECT_EQ(db.capacity(), 0u);
  EXPECT_EQ(db.data(), nullptr);
}

TEST(DeviceBuffer, FromHost) {
  DeviceBuffer<int> db;
  std::vector<int> v1(123), v2;
  db.from_host(v1);
  ASSERT_EQ(db.size(), v1.size());
  v2.resize(db.size());
  copyD2H(v2.data(), db.data(), v2.size());
  EXPECT_EQ(v1, v2);
}

TEST(DeviceBuffer, ResizePreserveContents) {
  DeviceBuffer<int> db;
  std::vector<int> v1(123), v2;
  db.from_host(v1);
  ASSERT_EQ(db.size(), v1.size());
  v2.resize(db.size());
  db.resize(256);
  copyD2H(v2.data(), db.data(), v2.size());
  EXPECT_EQ(v1, v2);
}

TEST(DeviceBuffer, Copy) {
  DeviceBuffer<int> db1, db2;
  std::vector<int> v1(123), v2;
  db1.from_host(v1);
  db2.copy(db1);
  ASSERT_EQ(db2.size(), v1.size());
  v2.resize(db2.size());
  copyD2H(v2.data(), db2.data(), v2.size());
  EXPECT_EQ(v1, v2);
}

TEST(DeviceBuffer, FromDevice) {
  DeviceBuffer<int> db1, db2;
  std::vector<int> v1(123), v2;
  db1.from_host(v1);
  db2.from_device(db1.data(), db1.size());
  ASSERT_EQ(db2.size(), v1.size());
  v2.resize(db2.size());
  copyD2H(v2.data(), db2.data(), v2.size());
  EXPECT_EQ(v1, v2);
}

TEST(DeviceBufferFail, Resize) {
  DeviceBuffer<int> db;
  size_t size = -1_uz;
  EXPECT_THROW(db.resize(size), CUDABadAlloc);
}

}  // namespace dali
