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

#include "input_provider.h"
#include <gtest/gtest.h>

namespace dali {
namespace test {
namespace detail {

bool succeed = false;

void cb_test(int) {
  succeed = true;
}

}  // namespace detail

class InputProviderTest : public ::testing::Test {
 public:
  void SetUp() final {}
  void TearDown() final {}
  void cb_test(int) {
    succeed_ = true;
  }
  static void cb_test_st(int) {
    succeed_static = true;
  }
  bool succeed_ = false;

  static bool succeed_static;
};

bool InputProviderTest::succeed_static = false;

TEST_F(InputProviderTest, LambdaTest) {
  bool succeed = false;
  InputProvider ip;
  ip.Register([&](int batch_size) { succeed = true; });
  ip.Notify(42);
  EXPECT_TRUE(succeed);
}

TEST_F(InputProviderTest, MemberFunctionTest) {
  InputProvider ip;
  ip.Register(std::bind(&InputProviderTest::cb_test, this, std::placeholders::_1));
  ip.Notify(42);
  EXPECT_TRUE(this->succeed_);
}

TEST_F(InputProviderTest, StaticMemberFunctionTest) {
  InputProvider ip;
  ip.Register(InputProviderTest::cb_test_st);
  ip.Notify(42);
  EXPECT_TRUE(InputProviderTest::succeed_static);
}

TEST_F(InputProviderTest, FreeFunctionTest) {
  InputProvider ip;
  ip.Register(detail::cb_test);
  ip.Notify(42);
  EXPECT_TRUE(detail::succeed);
}

}  // namespace test
}  // namespace dali