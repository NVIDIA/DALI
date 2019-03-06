// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/test/dali_operator_test.h"
#include "dali/test/dali_operator_test_utils.h"

namespace dali {
namespace testing {

TEST(ArgumentsCartesianTest, NoOp) {
  std::vector<testing::Arguments> args = {
      {{"arg0", std::string{"What is the answer to the ultimate question of life,"
                            " the universe and everything?"}},
       {"arg1", 42}},
      {}};
  auto c = cartesian(args);
  ASSERT_EQ(c.size(), args.size());
  ASSERT_EQ(c[0].size(), args[0].size());
  ASSERT_EQ(c[1].size(), args[1].size());
  ASSERT_EQ(c[0].at("arg0").GetValue<std::string>(), args[0].at("arg0").GetValue<std::string>());
  ASSERT_EQ(c[0].at("arg1").GetValue<int>(), args[0].at("arg1").GetValue<int>());
}

TEST(ArgumentsCartesianTest, Merge) {
  std::vector<testing::Arguments> args0 = {
      {{"arg00", 0}, {"arg01", 1}},
      {{"arg00", 2}, {"arg01", 3}},
  };

  std::vector<testing::Arguments> args1 = {
      {{"arg10", 0}, {"arg11", 1}},
      {{"arg10", 2}, {"arg11", 3}},
  };

  std::vector<testing::Arguments> args2 = {
      {{"arg00", 10}, {"arg01", 11}},
      {{"arg00", 12}, {"arg01", 13}},
  };
  ASSERT_EQ(cartesian(args0, args1).size(), 4);
  ASSERT_EQ(cartesian(args0, args1)[0].at("arg00").GetValue<int>(), 0);
  ASSERT_EQ(cartesian(args0, args1, args2).size(), 8);
  ASSERT_EQ(cartesian(args0, args1, args2)[0].at("arg00").GetValue<int>(), 10);
}

}  // namespace testing
}  // namespace dali
