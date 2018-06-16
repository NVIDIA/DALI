// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

typedef ::testing::Types<uint8,
                         int16,
                         int32,
                         int64,
                         float16,
                         float,
                         double,
                         bool> TestTypes;

TYPED_TEST_CASE(TypesTest, TestTypes);

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
