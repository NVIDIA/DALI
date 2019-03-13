// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/kernels/kernel.h"
#include "dali/kernels/static_switch.h"
#include "dali/kernels/type_tag.h"

namespace {
template <typename I, typename O, int C>
struct Functor {
  void operator()(int &calls, int i, int o, int c) {
    EXPECT_EQ(i, dali::TypeTag<I>::value);
    EXPECT_EQ(o, dali::TypeTag<O>::value);
    EXPECT_EQ(c, C);
    calls++;
  }
};

template <typename T>
struct StaticSwitch : testing::Test {};
}  // namespace

typedef testing::Types<int, float, double> MyTypes;
TYPED_TEST_SUITE(StaticSwitch, MyTypes);

TYPED_TEST(StaticSwitch, TypeSwitch) {
  using T = gtest_TypeParam_;
  int tag = -1;
  TYPE_SWITCH(dali::TypeTag<T>::value, dali::TypeTag, IType, (int, float, double),
    (
      tag = dali::TypeTag<IType>::value;
      EXPECT_TRUE((std::is_same<IType, T>::value)) << "TypeSwitch type mismatch";
    ),  // NOLINT
    GTEST_FAIL() << "Invalid type";
  );  // NOLINT

  EXPECT_EQ(dali::TypeTag<T>::value, tag)
    << "Tag mismatch - did type switch actually call anything?";
}

TEST(StaticSwitch, Nested) {
  int input_type = dali::TypeTag<float>();
  int output_type = dali::TypeTag<int64_t>();
  int calls = 0;
  for (int channels = 1; channels <= 4; channels++) {
    TYPE_SWITCH(input_type, dali::TypeTag, IType, (int, float, double, int64_t), (
        TYPE_SWITCH(output_type, dali::TypeTag, OType, (int, uint8_t, int64_t), (
            VALUE_SWITCH(channels, num_channels, (1, 2, 3, 4),
              (Functor<IType, OType, num_channels>()(calls, input_type, output_type, channels); ),
              FAIL() << "Unsupported value";
            )                                       // NOLINT
          ), FAIL() << "Unsupported output type";   // NOLINT
        )                                           // NOLINT
      ), FAIL() << "Unsupported input type";        // NOLINT
    )                                               // NOLINT
  }
  // check that the test functor was actually called
  EXPECT_EQ(calls, 4) << "Test functor was expected to be called 4 times";
}
