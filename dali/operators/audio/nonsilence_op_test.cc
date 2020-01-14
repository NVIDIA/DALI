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
#include <utility>
#include "dali/operators/audio/nonsilence_op.h"

namespace dali {
namespace testing {

TEST(NonsilenceOpTest, DetectNonsilenceRegion) {
  std::vector<float> t0 = {0, 0, 0, 0, 0, 1.5, -100, 1.5};
  EXPECT_EQ(detail::DetectNonsilenceRegion(make_cspan(t0), .5f), std::make_pair(5, 3));

  std::vector<float> t1 = {1.5, -100, 1.5, 0, 0, 0, 0};
  EXPECT_EQ(detail::DetectNonsilenceRegion(make_cspan(t1), .5f), std::make_pair(0, 3));

  std::vector<float> t2 = {0, 0, 0, 0, 0, 1.5, -100, -100, 1.5, 0, 0, 0, 0};
  EXPECT_EQ(detail::DetectNonsilenceRegion(make_cspan(t2), 1.5f), std::make_pair(5, 4));

  std::vector<int> t3 = {23, 62, 46, 12, 53};
  EXPECT_EQ(detail::DetectNonsilenceRegion(make_cspan(t3), 100).second, 0);

  std::vector<int64_t> t4 = {623, 45, 62, 46, 23};
  EXPECT_EQ(detail::DetectNonsilenceRegion(make_cspan(t4), 10L), std::make_pair(0, 5));
}

}  // namespace testing
}  // namespace dali
