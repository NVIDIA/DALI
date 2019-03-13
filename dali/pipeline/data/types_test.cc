// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/data/types.h"

#include <gtest/gtest.h>

#include <string>

#include "dali/test/dali_test.h"

namespace dali {

template <typename Type>
class TypesTest : public DALITest {
 public:
  // Returns the name of the current type
  std::string TypeName();
};

namespace {
  static constexpr size_t DUMMY_ARRAY_SIZE = 42;
}

#define TYPENAME_FUNC(type)                     \
  template <>                                   \
  std::string TypesTest<type>::TypeName() {     \
    return #type;                               \
  }                                             \

TYPENAME_FUNC(uint8);
TYPENAME_FUNC(int16);
TYPENAME_FUNC(int32);
TYPENAME_FUNC(int64);
TYPENAME_FUNC(float16);
TYPENAME_FUNC(float);
TYPENAME_FUNC(double);
TYPENAME_FUNC(bool);

template <>
std::string TypesTest<std::vector<uint8>>::TypeName() {
  return "list of uint8";
}

template <>
std::string TypesTest<std::array<std::vector<uint8>, DUMMY_ARRAY_SIZE>>::TypeName() {
  return "list of list of uint8";
}

typedef ::testing::Types<uint8,
                         int16,
                         int32,
                         int64,
                         float16,
                         float,
                         double,
                         bool,
                         std::vector<uint8>,
                         std::array<std::vector<uint8>, DUMMY_ARRAY_SIZE>
                         > TestTypes;

TYPED_TEST_SUITE(TypesTest, TestTypes);

TYPED_TEST(TypesTest, TestRegisteredType) {
  typedef TypeParam T;

  TypeInfo type;

  // Verify we start with no type
  ASSERT_EQ(type.name(), "NoType");
  ASSERT_EQ(type.size(), 0);

  type.SetType<T>();

  ASSERT_EQ(type.size(), sizeof(T));
  ASSERT_EQ(type.name(), this->TypeName());
}

}  // namespace dali
