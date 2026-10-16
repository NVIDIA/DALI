// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string>
#include "dali/core/format.h"
#include "dali/core/version.h"

namespace dali {

TEST(VersionTest, MacrosAreConsistent) {
  static_assert(DALI_VERSION_MAJOR >= 0);
  static_assert(DALI_VERSION_MINOR >= 0);
  static_assert(DALI_VERSION_PATCH >= 0);
  static_assert(DALI_VERSION == DALI_MAKE_VERSION(DALI_VERSION_MAJOR,
                                                  DALI_VERSION_MINOR,
                                                  DALI_VERSION_PATCH));
  static_assert(DALI_MAKE_VERSION(1, 2, 3) == 10203);
  static_assert(DALI_MAKE_VERSION(1, 2, 3) < DALI_MAKE_VERSION(1, 3, 0));

  std::string expected = make_string(DALI_VERSION_MAJOR, ".", DALI_VERSION_MINOR, ".",
                                     DALI_VERSION_PATCH, DALI_VERSION_SUFFIX);
  EXPECT_EQ(expected, DALI_VERSION_STRING);
}

}  // namespace dali
