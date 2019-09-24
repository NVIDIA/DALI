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
#include "dali/core/span.h"

namespace dali {
namespace test {

template <typename T>
void Verify(span<T> s, std::vector<T> vec) {
  ASSERT_EQ(vec.size(), s.size());
  for (int i = 0; i < s.size(); i++) {
    EXPECT_EQ(vec[i], s[i]);
  }
}


template <typename T>
void Verify(span<const T> s, std::vector<T> vec) {
  ASSERT_EQ(vec.size(), s.size());
  for (int i = 0; i < s.size(); i++) {
    EXPECT_EQ(vec[i], s[i]);
  }
}


TEST(SpanTest, cspan_test) {
  std::vector<int> v = {1, 2, 3, 4, 5};
  auto &ref = v;
  const auto &cref = v;
  Verify(make_cspan(v), v);
  Verify(make_cspan(ref), v);
  Verify(make_cspan(cref), v);
}

}  // namespace test
}  // namespace dali
