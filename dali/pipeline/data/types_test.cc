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
  ASSERT_EQ(type.name(), "NoType");
  ASSERT_EQ(type.size(), 0);

  type.SetType<T>();

  ASSERT_EQ(type.size(), sizeof(T));
  ASSERT_EQ(type.name(), this->TypeName());
}

#define GET_TYPE_INFO(Type) \
const auto &ANONYMIZE_VARIABLE(info) = TypeTable::GetTypeInfo(type2id<Type>::value)

TEST(TypeTableTest, BasicTypesLookup) {
  GET_TYPE_INFO(uint8_t);
  GET_TYPE_INFO(uint16_t);
  GET_TYPE_INFO(uint32_t);
  GET_TYPE_INFO(uint64_t);
  GET_TYPE_INFO(int8_t);
  GET_TYPE_INFO(int16_t);
  GET_TYPE_INFO(int32_t);
  GET_TYPE_INFO(int64_t);
  GET_TYPE_INFO(float16);
  GET_TYPE_INFO(float);
  GET_TYPE_INFO(double);
  GET_TYPE_INFO(bool);
  GET_TYPE_INFO(string);
  GET_TYPE_INFO(DALIImageType);
  GET_TYPE_INFO(DALIDataType);
  GET_TYPE_INFO(DALIInterpType);
  GET_TYPE_INFO(DALITensorLayout);
#ifdef DALI_BUILD_PROTO3
  GET_TYPE_INFO(TFUtil::Feature);
  GET_TYPE_INFO(std::vector<TFUtil::Feature>);
#endif
  GET_TYPE_INFO(std::vector<bool>);
  GET_TYPE_INFO(std::vector<int>);
  GET_TYPE_INFO(std::vector<std::string>);
  GET_TYPE_INFO(std::vector<float>);
}

}  // namespace dali
