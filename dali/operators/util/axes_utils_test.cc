// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>
#include "dali/operators/util/axes_utils.h"

namespace dali {
namespace kernels {

TEST(AxesUtils, CheckAxes) {
  EXPECT_NO_THROW(CheckAxes(span<const int>{}, 0));
  int axes_0[] = { 0 };
  int axes_01[] = { 0, 1 };
  int axes_2[] = { 2 };
  int axes_neg1[] = { -1 };
  int axes_neg2[] = { -2 };
  int axes_neg12[] = { -2, -1 };
  int axes_pos0_neg1[] = {0, -1};
  EXPECT_NO_THROW(CheckAxes(make_span(axes_0), 1));
  EXPECT_NO_THROW(CheckAxes(make_span(axes_01), 2));

  EXPECT_THROW(CheckAxes(make_span(axes_neg1), 1), std::out_of_range);
  EXPECT_THROW(CheckAxes(make_span(axes_neg2), 2), std::out_of_range);
  EXPECT_THROW(CheckAxes(make_span(axes_neg12), 2), std::out_of_range);
  EXPECT_THROW(CheckAxes(make_span(axes_pos0_neg1), 2), std::out_of_range);
  EXPECT_THROW(CheckAxes(make_span(axes_neg2), 1), std::out_of_range);
  EXPECT_THROW(CheckAxes(make_span(axes_2), 2), std::out_of_range);
}

TEST(AxesUtils, ProcessNegativeAxes) {
  auto test = [](span<int> axes, int ndim) {
    ProcessNegativeAxes(axes, ndim);
    CheckAxes(axes, ndim);
  };
  int axes_neg1[] = { -1 };
  int axes_neg2[] = { -2 };
  int axes_neg12[] = { -2, -1 };
  int axes_010[] = { 0, 1, 0 };
  int axes_pos0_neg1[] = {0, -1};
  int axes_neg121[] = {-2, -1, -2 };
  EXPECT_NO_THROW(test(make_span(axes_neg1), 1));
  EXPECT_NO_THROW(test(make_span(axes_neg2), 2));
  EXPECT_NO_THROW(test(make_span(axes_neg12), 2));

  EXPECT_THROW(test(make_span(axes_pos0_neg1), 1), std::invalid_argument);
  EXPECT_THROW(test(make_span(axes_010), 2), std::invalid_argument);
  EXPECT_THROW(test(make_span(axes_neg121), 2), std::invalid_argument);
}


}  // namespace kernels
}  // namespace dali
