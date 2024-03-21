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
#include "dali/core/source_location.h"
#include "dali/test/device_test.h"


namespace dali {
namespace test {

DALI_HOST_DEV int naive_strlen(const char *str) {
  int i = 0;
  while (str[i] != '\0') {
    i++;
  }
  return i;
}

DALI_HOST_DEV bool compare(const char *str1, const char *str2, int len) {
  for (int i = 0; i < len; i++) {
    if (str1[i] != str2[i]) {
      return false;
    }
  }
  return true;
}

DALI_HOST_DEV bool naive_contains(const char *haystack, const char *needle) {
  int haystack_len = naive_strlen(haystack);
  int needle_len = naive_strlen(needle);
  int search_start_pos = 0;
  for (int start = 0; start <= haystack_len - needle_len; start++) {
    if (compare(haystack + start, needle, needle_len)) {
      return true;
    }
  }
  return false;
}

DALI_HOST_DEV source_location GetDev(source_location loc = source_location::current()) {
  return loc;
}

DALI_HOST_DEV source_location IndirectDev() {
  return GetDev();
}

DEVICE_TEST(SourceLocationDev, CurrentLocationDevTest, dim3(1), dim3(1)) {
  source_location default_loc;
  printf("\"%s\":%d in \"%s\"\n", default_loc.source_file(), default_loc.line(),
         default_loc.function_name());
  DEV_ASSERT_EQ(default_loc.source_file()[0], '\0');
  DEV_ASSERT_EQ(default_loc.function_name()[0], '\0');
  DEV_ASSERT_EQ(default_loc.line(), 0);


  source_location current_loc = source_location::current();
  printf("\"%s\":%d in \"%s\"\n", current_loc.source_file(), current_loc.line(),
         current_loc.function_name());
  DEV_ASSERT_TRUE(naive_contains(current_loc.source_file(), "source_location_test.cu"));
  DEV_ASSERT_TRUE(naive_contains(current_loc.function_name(), "CurrentLocationDevTest"));
  DEV_ASSERT_NE(current_loc.line(), 0);


  auto returned_loc = IndirectDev();
  printf("\"%s\":%d in \"%s\"\n", returned_loc.source_file(), returned_loc.line(),
         returned_loc.function_name());
  DEV_ASSERT_TRUE(naive_contains(current_loc.source_file(), returned_loc.source_file()));
  DEV_ASSERT_TRUE(naive_contains(returned_loc.function_name(), "IndirectDev"));
  DEV_ASSERT_GT(current_loc.line(), returned_loc.line());
}

}  // namespace test
}  // namespace dali
