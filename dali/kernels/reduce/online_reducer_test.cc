// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/kernels/reduce/online_reducer.h"

namespace dali {
namespace kernels {

TEST(OnlineReducer, TestVsDoubleSum) {
  double ref_sum = 0;
  OnlineReducer<float, reductions::sum> r;
  r.reset();
  EXPECT_EQ(r.result(), 0);
  for (int i = 0; i < (1 << 24) - 1; i++) {
    float x = i * 1e-5f + 1;
    ref_sum += x;
    r.add(x);
  }
  double eps = ref_sum * 1e-7f;
  EXPECT_NEAR(r.result(), ref_sum, eps);
}

}  // namespace kernels
}  // namespace dali
