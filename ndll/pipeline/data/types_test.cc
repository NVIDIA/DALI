#include "ndll/pipeline/data/types.h"

#include <gtest/gtest.h>

#include <string>

#include "ndll/test/ndll_test.h"

namespace ndll {

template <typename Type>
class TypesTest : public NDLLTest {
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
TYPENAME_FUNC(int);
TYPENAME_FUNC(long);
TYPENAME_FUNC(long long);
TYPENAME_FUNC(float16);
TYPENAME_FUNC(float);
TYPENAME_FUNC(double);

typedef ::testing::Types<uint8,
                         int16,
                         int,
                         long,
                         long long,
                         float16,
                         float,
                         double> TestTypes;

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

} // namespace ndll
