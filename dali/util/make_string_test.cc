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
#include "dali/core/geom/vec.h"
#include "dali/util/make_string.h"

namespace dali {

TEST(MakeStringTest, default_delimiter) {
  auto str = make_string("jeden", 2, 3);
  ASSERT_EQ(str, "jeden 2 3");
}

TEST(MakeStringTest, custom_delimiter) {
  auto str = make_string_delim("a custom delimiter", "jeden", 2, 3);
  ASSERT_EQ(str, "jedena custom delimiter2a custom delimiter3");
}

TEST(MakeStringTest, no_arguments) {
  auto str = make_string();
  ASSERT_EQ(str, "");
}

TEST(MakeStringTest, one_argument) {
  auto str = make_string("d[-_-]b");
  ASSERT_EQ(str, "d[-_-]b");
}

TEST(MakeStringTest, only_delimiter) {
  auto str = make_string_delim(">.<");
  ASSERT_EQ(str, "");
}

TEST(MakeStringTest, delimiter_and_one_argument) {
  auto str = make_string_delim("it really doesn't matter what's in here", "( . Y . )");
  ASSERT_EQ(str, "( . Y . )");
}


}  // namespace dali
