// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include  "dali/core/source_location.h"



namespace dali {
namespace test {


source_location get(source_location loc = source_location::current()) {
  return loc;
}

source_location indirect() {
  return get();
}

void TestFunction() {
  source_location current_loc = source_location::current();

  std::cout << current_loc.source_file() << " " << current_loc.function_name() << " "
            << current_loc.line() << std::endl;

  ASSERT_NE(strstr(current_loc.source_file(), "source_location_test.cc"), nullptr);
  ASSERT_NE(strstr(current_loc.function_name(), "TestFunction"), nullptr);
  ASSERT_NE(current_loc.line(), 0);

  auto returned_loc = indirect();

  std::cout << returned_loc.source_file() << " " << returned_loc.function_name() << " "
            << returned_loc.line() << std::endl;

  ASSERT_STREQ(returned_loc.source_file(), current_loc.source_file());
  ASSERT_NE(strstr(returned_loc.function_name(), "indirect"), nullptr);
  ASSERT_GT(current_loc.line(), returned_loc.line());
}

TEST(SourceLocation, CurrentLocationTest) {
  source_location default_loc;

  ASSERT_STREQ(default_loc.source_file(), "");
  ASSERT_STREQ(default_loc.function_name(), "");
  ASSERT_EQ(default_loc.line(), 0);

  TestFunction();
}

}  // namespace test
}  // namespace dali
