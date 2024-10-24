// Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/operator/builtin/caching_list.h"
#include <gtest/gtest.h>
#include <utility>

namespace dali::test {

namespace {

template<typename T>
struct TestType {
  using element_type = T;
  T val;

  bool operator==(const T &other) const {
    return other == val;
  }
};

}  // namespace


TEST(CachingListTest, ProphetTest) {
  CachingList<TestType<int>> cl;

  auto push = [&](int val) {
      auto elem = cl.GetEmpty();
      elem->val = val;
      cl.PushBack(std::move(elem));
  };

  ASSERT_THROW(cl.PeekProphet(), std::out_of_range);
  push(6);
  EXPECT_EQ(cl.PeekProphet(), 6);
  push(9);
  EXPECT_EQ(cl.PeekProphet(), 6);
  cl.AdvanceProphet();
  EXPECT_EQ(cl.PeekProphet(), 9);
  push(13);
  EXPECT_EQ(cl.PeekProphet(), 9);
  cl.AdvanceProphet();
  EXPECT_EQ(cl.PeekProphet(), 13);
  push(42);
  EXPECT_EQ(cl.PeekProphet(), 13);
  push(69);
  EXPECT_EQ(cl.PeekProphet(), 13);
  cl.AdvanceProphet();
  EXPECT_EQ(cl.PeekProphet(), 42);
  cl.AdvanceProphet();
  EXPECT_EQ(cl.PeekProphet(), 69);
  cl.AdvanceProphet();
  ASSERT_THROW(cl.PeekProphet(), std::out_of_range);
  push(666);
  EXPECT_EQ(cl.PeekProphet(), 666);
  push(1337);
  EXPECT_EQ(cl.PeekProphet(), 666);
  cl.AdvanceProphet();
  EXPECT_EQ(cl.PeekProphet(), 1337);
  cl.AdvanceProphet();
  ASSERT_THROW(cl.PeekProphet(), std::out_of_range);
  push(1234);
  EXPECT_EQ(cl.PeekProphet(), 1234);
  push(4321);
  EXPECT_EQ(cl.PeekProphet(), 1234);
  cl.AdvanceProphet();
  EXPECT_EQ(cl.PeekProphet(), 4321);
  cl.AdvanceProphet();
  ASSERT_THROW(cl.PeekProphet(), std::out_of_range);
  ASSERT_THROW(cl.AdvanceProphet(), std::out_of_range);
}
}  // namespace dali::test
