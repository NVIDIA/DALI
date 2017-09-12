#include "ndll/pipeline/data/tensor.h"

#include <gtest/gtest.h>

#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/buffer.h"

namespace ndll {

template <typename Backend>
class TensorTest : public ::testing::Test {
public:
  void SetUp() {
    rand_gen_.seed(time(nullptr));
  }

  void TearDown() {

  }

  int RandInt(int a, int b) {
    return std::uniform_int_distribution<>(a, b)(rand_gen_);
  }

  template <typename T>
  auto RandReal(int a, int b) -> T {
    return std::uniform_real_distribution<>(a, b)(rand_gen_);
  }

  vector<Index> GetRandShape() {
    int dims = this->RandInt(0, 5);
    vector<Index> shape(dims, 0);
    for (auto &val : shape) {
      // Dims cannot be of size 0
      val = this->RandInt(1, 50);
    }
    return shape;
  }
  
protected:
  std::mt19937 rand_gen_;
};

typedef ::testing::Types<CPUBackend,
                         PinnedCPUBackend,
                         GPUBackend> Backends;
TYPED_TEST_CASE(TensorTest, Backends);

TYPED_TEST(TensorTest, TestResize) {
  try {
    Tensor<TypeParam> tensor;

    // Get shape
    vector<Index> shape = this->GetRandShape();
    tensor.Resize(shape);

    // Verify the settings
    ASSERT_NE(tensor.template data<float>(), nullptr);
    ASSERT_EQ(tensor.size(), Product(shape));
    ASSERT_EQ(tensor.ndim(), shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      ASSERT_EQ(tensor.dim(i), shape[i]);      
    }
    
  } catch (NDLLException &e) {
    FAIL() << e.what();
  }
}

TYPED_TEST(TensorTest, TestResizeScalar) {
  try {
    Tensor<TypeParam> tensor;

    // Get shape
    vector<Index> shape = {};
    tensor.Resize(shape);
       
    // Verify the settings
    ASSERT_NE(tensor.template data<float>(), nullptr);
    ASSERT_EQ(tensor.size(), Product(shape));
    ASSERT_EQ(tensor.ndim(), shape.size());
  } catch (NDLLException &e) {
    FAIL() << e.what();
  }
}

} // namespace ndll
