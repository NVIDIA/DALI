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
#include "dali/operators/audio/mfcc/mfcc.h"

namespace dali {
namespace detail {
namespace test {

void check_lifter_coeffs(LifterCoeffs& coeffs, double lifter, int64_t length) {
  ASSERT_EQ(length, coeffs.size());
  auto coeffs_data = coeffs.data();
  ASSERT_NE(nullptr, coeffs_data);
  for (int64_t i = 0; i < length; i++) {
    float expected = 1.0 + 0.5 * lifter * std::sin((i + 1) * M_PI / lifter);
    EXPECT_FLOAT_EQ(expected, coeffs_data[i]);
  }
}

TEST(LifterCoeffs, correctness) {
  LifterCoeffs coeffs;

  auto lifter = 0.0f;
  coeffs.Calculate(10, lifter);
  ASSERT_TRUE(coeffs.empty());

  lifter = 1.234f;
  coeffs.Calculate(10, lifter);
  check_lifter_coeffs(coeffs, lifter, 10);

  coeffs.Calculate(20, lifter);
  check_lifter_coeffs(coeffs, lifter, 20);

  lifter = 2.234f;
  coeffs.Calculate(10, lifter);
  check_lifter_coeffs(coeffs, lifter, 10);

  coeffs.Calculate(5, lifter);
  check_lifter_coeffs(coeffs, lifter, 10);
}

}  // namespace test
}  // namespace detail
}  // namespace dali
