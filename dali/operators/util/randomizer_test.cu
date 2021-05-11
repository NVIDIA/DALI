// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/util/randomizer.cuh"
#include "dali/test/device_test.h"

namespace dali {
namespace test {

DEVICE_TEST(uniform_dist, default_ctor, 110, 1024) {
  uint64_t seed = 12345;
  curandState state;
  curand_init(seed, 0, 0, &state);
  curand_uniform_dist<float> dist_f;
  curand_uniform_dist<double> dist_f64;
  for (int i = 0; i < 3; i++) {
    auto n1 = dist_f(&state);
    DEV_EXPECT_GE(n1, 0.0f);
    DEV_EXPECT_LT(n1, 1.0f);

    auto n2 = dist_f64(&state);
    DEV_EXPECT_GE(n2, 0.0);
    DEV_EXPECT_LT(n2, 1.0);
  }
}

DEVICE_TEST(uniform_dist, custom_range, 110, 1024) {
  uint64_t seed = 12345;
  curandState state;
  curand_init(seed, 0, 0, &state);
  curand_uniform_dist<float> dist_f(-0.5, 0.5);
  curand_uniform_dist<double> dist_f64(-0.5, 0.5);
  for (int i = 0; i < 3; i++) {
    auto n1 = dist_f(&state);
    DEV_EXPECT_GE(n1, -0.5f);
    DEV_EXPECT_LT(n1, 0.5f);

    auto n2 = dist_f64(&state);
    DEV_EXPECT_GE(n2, -0.5);
    DEV_EXPECT_LT(n2, 0.5);
  }
}


}  // namespace test
}  // namespace dali
