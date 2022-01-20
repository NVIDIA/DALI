// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/reader/loader/webdataset_loader.h"

namespace dali::detail::wds::test {

TEST(WebdatasetLoaderTest, ParseIndexStringTest) {
  EXPECT_EQ(ParseIndexVersion("v1.0"), 1000);
  EXPECT_EQ(ParseIndexVersion("v1.5"), 1005);
  EXPECT_EQ(ParseIndexVersion("v1.10"), 1010);
  EXPECT_EQ(ParseIndexVersion("v5.0"), 5000);
  EXPECT_EQ(ParseIndexVersion("v5.5"), 5005);
  EXPECT_EQ(ParseIndexVersion("v5.10"), 5010);
  EXPECT_EQ(ParseIndexVersion("v10.0"), 10000);
  EXPECT_EQ(ParseIndexVersion("v10.5"), 10005);
  EXPECT_EQ(ParseIndexVersion("v10.10"), 10010);
  EXPECT_EQ(ParseIndexVersion("v11.0"), 11000);
  EXPECT_EQ(ParseIndexVersion("v11.5"), 11005);
  EXPECT_EQ(ParseIndexVersion("v11.10"), 11010);
}

TEST(WebdatasetLoaderTest, VerifyIndexVersionStringTest) {
  EXPECT_TRUE(VerifyIndexVersionString("v0.0"));
  EXPECT_TRUE(VerifyIndexVersionString("v00.00"));
  EXPECT_FALSE(VerifyIndexVersionString("anystring"));
  EXPECT_FALSE(VerifyIndexVersionString("v."));
  EXPECT_FALSE(VerifyIndexVersionString("v.1"));
  EXPECT_FALSE(VerifyIndexVersionString("v1."));
  EXPECT_FALSE(VerifyIndexVersionString("v 1.0"));
  EXPECT_FALSE(VerifyIndexVersionString("v.1.0"));
  EXPECT_FALSE(VerifyIndexVersionString("v1..0"));
  EXPECT_FALSE(VerifyIndexVersionString("vv1.0"));
  EXPECT_FALSE(VerifyIndexVersionString("1.0"));
  EXPECT_FALSE(VerifyIndexVersionString("v1"));
  EXPECT_FALSE(VerifyIndexVersionString("v1.0."));
}

}  // namespace dali::detail::wds::test
