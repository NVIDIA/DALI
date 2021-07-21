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
#include <utility>
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

template <typename T>
auto GetTypeInfo() {
  return std::make_pair(type2id<T>::value, &TypeTable::GetTypeInfo(type2id<T>::value));
}

static std::pair<DALIDataType, const TypeInfo *> g_types[] = {
  GetTypeInfo<uint8_t>(),
  GetTypeInfo<int8_t>(),
  GetTypeInfo<uint16_t>(),
  GetTypeInfo<int16_t>(),
  GetTypeInfo<uint32_t>(),
  GetTypeInfo<int32_t>(),
  GetTypeInfo<uint64_t>(),
  GetTypeInfo<int64_t>(),
  GetTypeInfo<bool>(),
  GetTypeInfo<float>(),
  GetTypeInfo<double>(),
  GetTypeInfo<string>(),
  GetTypeInfo<DALIDataType>()
};


TEST(TypeTableTest, BasicTypesLookup) {
  for (auto &info : g_types) {
    ASSERT_NE(info.second, nullptr);
    EXPECT_EQ(info.first, info.second->id());
  }
}

#define TYPENAME_FUNC(type, type_name)           \
  template <>                                   \
  std::string TypesTest<type>::TypeName() {     \
    return #type_name;                               \
  }                                             \

TYPENAME_FUNC(uint8_t, uint8);
TYPENAME_FUNC(uint16_t, uint16);
TYPENAME_FUNC(uint32_t, uint32);
TYPENAME_FUNC(uint64_t, uint64);
TYPENAME_FUNC(int8_t, int8);
TYPENAME_FUNC(int16_t, int16);
TYPENAME_FUNC(int32_t, int32);
TYPENAME_FUNC(int64_t, int64);
TYPENAME_FUNC(float16, float16);
TYPENAME_FUNC(float, float);
TYPENAME_FUNC(double, double);
TYPENAME_FUNC(bool, bool);

template <>
std::string TypesTest<std::vector<uint8>>::TypeName() {
  return "list of uint8";
}

template <>
std::string TypesTest<std::array<std::vector<uint8>, DUMMY_ARRAY_SIZE>>::TypeName() {
  return "list of list of uint8";
}

typedef ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t,
                         float16, float, double, bool, std::vector<uint8>,
                         std::array<std::vector<uint8>, DUMMY_ARRAY_SIZE>>
    TestTypes;

TYPED_TEST_SUITE(TypesTest, TestTypes);

TYPED_TEST(TypesTest, TestRegisteredType) {
  typedef TypeParam T;

  TypeInfo type;

  // Verify we start with no type
  ASSERT_EQ(type.name(), "no_type");
  ASSERT_EQ(type.size(), 0);

  type.SetType<T>();

  ASSERT_EQ(type.size(), sizeof(T));
  ASSERT_EQ(type.name(), this->TypeName());
}

TEST(ListTypeNames, ListTypeNames) {
  auto str1 = ListTypeNames<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t,
                            int64_t, float, double, float16>();
  auto expected_str1 =
      "uint8, int8, uint16, int16, uint32, int32, uint64, int64, float, double, float16";
  ASSERT_EQ(str1, expected_str1);
}

}  // namespace dali
