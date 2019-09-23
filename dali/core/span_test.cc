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

TEST(SpanTest, ContainerTest) {
  std::vector<int> vec = {1, 2, 3, 4, 5, 6};
  span<int> sp{vec};

  ASSERT_EQ(sp.size(), vec.size());
  for (size_t i = 0; i < vec.size(); i++) {
    EXPECT_EQ(sp[i], vec[i]);
  }
}


TEST(SpanTest, ImplicitConversion) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    span<int> sp{vec};
    auto func = [&sp](span<const int> _sp) -> bool {
        if (sp.size() != _sp.size()) return false;
        for (int i = 0; i < sp.size(); i++) {
            if (sp[i] != _sp[i]) return false;
        }
        return true;
    };

    EXPECT_TRUE(func(vec));
}

}  // namespace dali
