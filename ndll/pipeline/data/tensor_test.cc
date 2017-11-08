#include "ndll/pipeline/data/tensor.h"

#include <gtest/gtest.h>

#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/buffer.h"
#include "ndll/test/ndll_test.h"

namespace ndll {

template <typename Backend>
class TensorTest : public NDLLTest {
public:
  vector<Dims> GetRandShapeList() {
    int num_tensor = this->RandInt(1, 128);
    vector<Dims> shape(num_tensor);
    for (int i = 0; i < num_tensor; ++i) {
      int dims = this->RandInt(1, 3);
      vector<Index> tensor_shape(dims, 0);
      for (int j = 0; j < dims; ++j) {
        tensor_shape[j] = this->RandInt(1, 512);
      }
      shape[i] = tensor_shape;
    }
    return shape;
  }
  
  vector<Index> GetRandShape() {
    int dims = this->RandInt(1, 5);
    vector<Index> shape(dims, 0);
    for (auto &val : shape) {
      val = this->RandInt(1, 32);
    }
    return shape;
  }
  
protected:
};

typedef ::testing::Types<CPUBackend,
                         GPUBackend> Backends;
TYPED_TEST_CASE(TensorTest, Backends);

// Sharing data from a raw pointer resets a Tensor to
// and invalid state (no type). To get to a valid state
// we can aquire a type and size in the following orders:
//
// type -> size (bytes) : getting size triggers allocation
// size -> type (bytes) : getting type triggers allocation
// bytes -> type -> size : shares data, gets type (zero size), gets size
// (may or may not allocate)
// bytes -> size -> type : shares data, gets size (no type), gets type
// (may or may not allocate)
//
// The following tests attempt to verify the correct behavior for
// all of these cases

TYPED_TEST(TensorTest, TestGetTypeSizeBytes) {
  Tensor<TypeParam> t;

  // Give the tensor a type
  t.template mutable_data<float>();

  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.raw_data(), nullptr);

  // Give the tensor a size. This
  // should trigger an allocation
  auto shape = this->GetRandShape();
  auto size = Product(shape);
  t.Resize(shape);

  // Validate the internals
  ASSERT_NE(t.raw_data(), nullptr);
  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.shape(), shape);
  ASSERT_EQ(t.nbytes(), size*sizeof(float));
  ASSERT_TRUE(IsType<float>(t.type()));
}

TYPED_TEST(TensorTest, TestGetSizeTypeBytes) {
  Tensor<TypeParam> t;

  // Give the tensor a size
  auto shape = this->GetRandShape();
  auto size = Product(shape);
  t.Resize(shape);

  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.shape(), shape);
  ASSERT_EQ(t.nbytes(), 0);

  // Give the tensor a type. This should
  // trigger an allocation
  t.template mutable_data<float>();

  // Validate the internals
  ASSERT_NE(t.raw_data(), nullptr);
  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.shape(), shape);
  ASSERT_EQ(t.nbytes(), size*sizeof(float));
  ASSERT_TRUE(IsType<float>(t.type()));
}

TYPED_TEST(TensorTest, TestGetBytesTypeSizeNoAlloc) {
  Tensor<TypeParam> t;

  // Get an allocation
  auto shape = this->GetRandShape();
  auto size = Product(shape);
  float *ptr = new float[size];

  // Wrap the allocation
  t.ShareData(ptr, size*sizeof(float));

  // Verify internals
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), vector<Index>{});
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_TRUE(t.shares_data());
  
  // Give the Tensor a type
  t.template mutable_data<int16>();

  ASSERT_EQ(t.raw_data(), ptr);
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), vector<Index>{});
  ASSERT_TRUE(IsType<int16>(t.type()));
  ASSERT_TRUE(t.shares_data());
  
  // Give the Tensor a size - should not trigger allocation
  t.Resize(shape);

  ASSERT_EQ(t.raw_data(), ptr);
  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), size*sizeof(int16));
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(IsType<int16>(t.type()));
  ASSERT_TRUE(t.shares_data());
}

TYPED_TEST(TensorTest, TestGetBytesTypeSizeAlloc) {
  Tensor<TypeParam> t;

  // Get an allocation
  auto shape = this->GetRandShape();
  auto size = Product(shape);
  float *ptr = new float[size];

  // Wrap the allocation
  t.ShareData(ptr, size*sizeof(float));

  // Verify internals
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), vector<Index>{});
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_TRUE(t.shares_data());
  
  // Give the Tensor a type
  t.template mutable_data<double>();

  ASSERT_EQ(t.raw_data(), ptr);
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), vector<Index>{});
  ASSERT_TRUE(IsType<double>(t.type()));
  ASSERT_TRUE(t.shares_data());
  
  // Give the Tensor a size - should not trigger allocation
  t.Resize(shape);

  ASSERT_NE(t.raw_data(), ptr);
  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), size*sizeof(double));
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(IsType<double>(t.type()));
  ASSERT_FALSE(t.shares_data());
}

TYPED_TEST(TensorTest, TestGetBytesSizeTypeNoAlloc) {
  Tensor<TypeParam> t;

  // Get an allocation
  auto shape = this->GetRandShape();
  auto size = Product(shape);
  float *ptr = new float[size];

  // Wrap the allocation
  t.ShareData(ptr, size*sizeof(float));

  // Verify internals
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), vector<Index>{});
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_TRUE(t.shares_data());

  // Give the Tensor a size
  t.Resize(shape);

  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(t.shares_data());
  
  // Give the Tensor a type
  t.template mutable_data<int16>();

  ASSERT_EQ(t.raw_data(), ptr);
  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), size*sizeof(int16));
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(IsType<int16>(t.type()));
  ASSERT_TRUE(t.shares_data());
}

TYPED_TEST(TensorTest, TestGetBytesSizeTypeAlloc) {
  Tensor<TypeParam> t;

  // Get an allocation
  auto shape = this->GetRandShape();
  auto size = Product(shape);
  float *ptr = new float[size];

  // Wrap the allocation
  t.ShareData(ptr, size*sizeof(float));

  // Verify internals
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), vector<Index>{});
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_TRUE(t.shares_data());

  // Give the Tensor a size
  t.Resize(shape);

  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(t.shares_data());
  
  // Give the Tensor a type
  t.template mutable_data<double>();

  ASSERT_NE(t.raw_data(), ptr);
  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), size*sizeof(double));
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(IsType<double>(t.type()));
  ASSERT_FALSE(t.shares_data());
}


// Sharing data from a TensorList puts a Tensor into
// a valid state (it has a type), as we enforce that
// the input TensorList has a valid type.
TYPED_TEST(TensorTest, TestShareData) {
  TensorList<TypeParam> tl;
  auto shape = this->GetRandShapeList();
  tl.Resize(shape);
  tl.template mutable_data<float>();

  // Create a tensor and wrap each tensor from the list
  Tensor<TypeParam> tensor;
  int num_tensor = tl.ntensor();
  for (int i = 0; i < num_tensor; ++i) {
    tensor.ShareData(&tl, i);

    // Verify the internals
    ASSERT_TRUE(tensor.shares_data());
    ASSERT_EQ(tensor.raw_data(), tl.raw_tensor(i));
    ASSERT_EQ(tensor.type(), tl.type());
    ASSERT_EQ(tensor.shape(), tl.tensor_shape(i));

    Index size = Product(tl.tensor_shape(i));
    ASSERT_EQ(tensor.size(), size);
    ASSERT_EQ(tensor.nbytes(), size*sizeof(float));
  }
}

TYPED_TEST(TensorTest, TestResize) {
  Tensor<TypeParam> tensor;

  // Get shape
  vector<Index> shape = this->GetRandShape();
  tensor.Resize(shape);

  // Verify the settings
  ASSERT_NE(tensor.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tensor.size(), Product(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(tensor.dim(i), shape[i]);      
  }
}

TYPED_TEST(TensorTest, TestMultipleResize) {
  Tensor<TypeParam> tensor;

  int num = this->RandInt(2, 20);
  for (int i = 0; i < num; ++i) {
    // Get shape
    vector<Index> shape = this->GetRandShape();
    tensor.Resize(shape);
      
    // Verify the settings
    ASSERT_NE(tensor.template mutable_data<float>(), nullptr);
    ASSERT_EQ(tensor.size(), Product(shape));
    ASSERT_EQ(tensor.ndim(), shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      ASSERT_EQ(tensor.dim(i), shape[i]);      
    }
  }
}

TYPED_TEST(TensorTest, TestResizeScalar) {
  Tensor<TypeParam> tensor;

  // Get shape
  vector<Index> shape = {1};
  tensor.Resize(shape);
       
  // Verify the settings
  ASSERT_NE(tensor.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tensor.size(), Product(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
}

TYPED_TEST(TensorTest, TestResizeZeroSize) {
  Tensor<TypeParam> tensor;

  // Get shape
  vector<Index> shape = {};
  tensor.Resize(shape);
       
  // Verify the settings
  ASSERT_EQ(tensor.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tensor.size(), Product(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
}

TYPED_TEST(TensorTest, TestTypeChange) {
  Tensor<TypeParam> tensor;
  
  // Get shape
  vector<Index> shape = this->GetRandShape();
  tensor.Resize(shape);
  
  // Verify the settings
  ASSERT_NE(tensor.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tensor.size(), Product(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(tensor.dim(i), shape[i]);      
  }

  // Save the pointer
  const void *ptr = tensor.raw_data();
  size_t nbytes = tensor.nbytes();
  
  // Change the type of the buffer
  tensor.template mutable_data<int>();

  // Verify the settings
  ASSERT_EQ(tensor.size(), Product(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(tensor.dim(i), shape[i]);      
  }

  // No re-allocation should have occured
  ASSERT_EQ(ptr, tensor.raw_data());
  ASSERT_EQ(nbytes, tensor.nbytes());
  
  // Change the type to a smaller type
  tensor.template mutable_data<uint8>();

  // Verify the settings
  ASSERT_EQ(tensor.size(), Product(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(tensor.dim(i), shape[i]);      
  }

  // No re-allocation should have occured
  ASSERT_EQ(ptr, tensor.raw_data());
  ASSERT_EQ(nbytes / sizeof(float) * sizeof(uint8), tensor.nbytes());
  
  // Change the type to a larger type
  tensor.template mutable_data<double>();

  // Verify the settings
  ASSERT_EQ(tensor.size(), Product(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(tensor.dim(i), shape[i]);      
  }

  // The memory should have been re-allocated
  ASSERT_NE(ptr, tensor.raw_data());
  ASSERT_EQ(nbytes / sizeof(float) * sizeof(double), tensor.nbytes());
}

} // namespace ndll
