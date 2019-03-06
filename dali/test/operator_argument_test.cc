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
#include "dali/test/operator_argument.h"

namespace dali {
namespace testing {

TEST(TestOpArg, Constructors) {
  {
    TestOpArg arg = 42;
    EXPECT_NO_THROW(EXPECT_EQ(arg.GetValue<int>(), 42));
  }
  {
    TestOpArg arg = 5u;
    EXPECT_NO_THROW(EXPECT_EQ(arg.GetValue<unsigned>(), 5u));
  }
  {
    TestOpArg arg = 'c';
    EXPECT_NO_THROW(EXPECT_EQ(arg.GetValue<char>(), 'c'));
  }
  {
    TestOpArg arg = 5.25f;
    EXPECT_NO_THROW(EXPECT_EQ(arg.GetValue<float>(), 5.25f));
  }
  {
    TestOpArg arg = "hello, world!";
    EXPECT_NO_THROW(EXPECT_EQ(arg.GetValue<std::string>(), "hello, world!"));
  }
  {
    const char *txt = "hello, world!";
    TestOpArg arg = txt;
    EXPECT_NO_THROW(EXPECT_EQ(arg.GetValue<std::string>(), "hello, world!"));
  }
}

TEST(TestOpArg, BadCast) {
  {
    TestOpArg arg = 42.0f;
    EXPECT_NO_THROW(arg.GetValue<float>());
    EXPECT_THROW(arg.GetValue<double>(), std::bad_cast);
    EXPECT_THROW(arg.GetValue<int>(), std::bad_cast);
  }
  {
    TestOpArg arg = static_cast<uint8_t>(5);
    EXPECT_NO_THROW(arg.GetValue<uint8_t>());
    EXPECT_THROW(arg.GetValue<int>(), std::bad_cast);
    EXPECT_THROW(arg.GetValue<unsigned>(), std::bad_cast);
  }
}

}  // namespace testing
}  // namespace dali
