// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include <dali/test/argument_key.h>
#include <gtest/gtest.h>

namespace dali {
namespace testing {

TEST(ArgumentKeyTest, ArgumentKeyTest) {
  ArgumentKey argk("arg");
  EXPECT_EQ("arg", argk.arg_name());
  EXPECT_EQ("", argk.node_name());

  ArgumentKey opargk("op", "arg");
  EXPECT_EQ("arg", opargk.arg_name());
  EXPECT_EQ("op", opargk.node_name());
}


TEST(ArgumentKeyTest, ArgumentOperatorLessTest) {
  ArgumentKey a1("arg1");
  ArgumentKey a2("arg2");

  ArgumentKey a3("A", "X");
  ArgumentKey a4("A", "Y");
  ArgumentKey a5("B", "X");

  EXPECT_TRUE(a1 < a2);
  EXPECT_FALSE(a2 < a1);
  EXPECT_TRUE(a3 < a4);
  EXPECT_FALSE(a4 < a3);
  EXPECT_TRUE(a4 < a5);
  EXPECT_FALSE(a5 < a4);
}

}  // namespace testing
}  // namespace dali
