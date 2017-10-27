#include "ndll/pipeline/data/sample.h"

#include <gtest/gtest.h>

#include "ndll/pipeline/data/batch.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/buffer.h"
#include "ndll/test/ndll_main_test.h"

namespace ndll {

template <typename Backend>
class SampleTest : public NDLLTest {
public:
  vector<Dims> GetRandBatchShape() {
    int batch_size = this->RandInt(1, 128);
    vector<Dims> shape(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      int dims = this->RandInt(0, 3);
      vector<Index> sample_shape(dims, 0);
      for (int j = 0; j < dims; ++j) {
        sample_shape[j] = this->RandInt(1, 512);
      }
      shape[i] = sample_shape;
    }
    return shape;
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
};

typedef ::testing::Types<CPUBackend,
                         GPUBackend> Backends;
TYPED_TEST_CASE(SampleTest, Backends);

TYPED_TEST(SampleTest, TestResize) {
  Sample<TypeParam> sample;

  // Get shape
  vector<Index> shape = this->GetRandShape();
  sample.Resize(shape);

  // Verify the settings
  ASSERT_NE(sample.template mutable_data<float>(), nullptr);
  ASSERT_EQ(sample.size(), Product(shape));
  ASSERT_TRUE(sample.owned());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(sample.shape()[i], shape[i]);
  }
}

TYPED_TEST(SampleTest, TestMultipleResize) {
  Sample<TypeParam> sample;

  int num = this->RandInt(2, 20);
  for (int i = 0; i < num; ++i) {
    // Get shape
    vector<Index> shape = this->GetRandShape();
    sample.Resize(shape);
      
    // Verify the settings
    ASSERT_NE(sample.template mutable_data<float>(), nullptr);
    ASSERT_EQ(sample.size(), Product(shape));
    ASSERT_TRUE(sample.owned());
    for (size_t i = 0; i < shape.size(); ++i) {
      ASSERT_EQ(sample.shape()[i], shape[i]);      
    }
  }
}

TYPED_TEST(SampleTest, TestGetSample) {
  // Setup batch of samples
  Batch<TypeParam> batch;
  vector<Dims> shape = this->GetRandBatchShape();
  int batch_size = shape.size();
  batch.Resize(shape);
  batch.template mutable_data<float>();

  // Wrap a sample
  int sample_idx = this->RandInt(0, batch_size-1);
  Sample<TypeParam> sample(&batch, sample_idx);

  // Check the dims, size, etc.
  ASSERT_EQ(sample.size(), Product(batch.sample_shape(sample_idx)));
  ASSERT_EQ(sample.type(), batch.type());
  ASSERT_EQ(sample.shape(), batch.sample_shape(sample_idx));
  ASSERT_EQ(sample.nbytes(),
      Product(batch.sample_shape(sample_idx))*batch.type().size());
  ASSERT_EQ(sample.template mutable_data<float>(), batch.template sample<float>(sample_idx));
  ASSERT_EQ(sample.owned(), false);

  // Now resize the data
  vector<Index> new_shape = this->GetRandShape();
  sample.Resize(new_shape);

  // Verify the settings
  ASSERT_NE(sample.template mutable_data<float>(), nullptr);
  ASSERT_EQ(sample.size(), Product(new_shape));
  ASSERT_TRUE(sample.owned());
  for (size_t i = 0; i < new_shape.size(); ++i) {
    ASSERT_EQ(sample.shape()[i], new_shape[i]);
  }
}

TYPED_TEST(SampleTest, TestTypeChange) {
  Sample<TypeParam> sample;

  // Get shape
  vector<Index> shape = this->GetRandShape();
  sample.Resize(shape);

  // Verify the settings
  ASSERT_NE(sample.template mutable_data<float>(), nullptr);
  ASSERT_EQ(sample.size(), Product(shape));
  ASSERT_TRUE(sample.owned());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(sample.shape()[i], shape[i]);
  }

  // Save the pointer
  const void *ptr = sample.raw_data();
  size_t nbytes = sample.nbytes();

  // Change the data type
  sample.template mutable_data<int>();

  // Verify the settings
  ASSERT_EQ(sample.size(), Product(shape));
  ASSERT_TRUE(sample.owned());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(sample.shape()[i], shape[i]);
  }

  // No allocation should have occured
  ASSERT_EQ(ptr, sample.raw_data());
  ASSERT_EQ(nbytes, sample.nbytes());

  // Change the data type to something smaller
  sample.template mutable_data<uint8>();

  // Verify the settings
  ASSERT_EQ(sample.size(), Product(shape));
  ASSERT_TRUE(sample.owned());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(sample.shape()[i], shape[i]);
  }

  // No allocation should have occured
  ASSERT_EQ(ptr, sample.raw_data());
  ASSERT_EQ(nbytes / sizeof(float) * sizeof(uint8), sample.nbytes());

  // Change the data type to something bigger
  sample.template mutable_data<double>();

  // Verify the settings
  ASSERT_EQ(sample.size(), Product(shape));
  ASSERT_TRUE(sample.owned());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(sample.shape()[i], shape[i]);
  }

  // Allocation should have occured
  ASSERT_NE(ptr, sample.raw_data());
  ASSERT_EQ(nbytes / sizeof(float) * sizeof(double), sample.nbytes());
}

} // namespace ndll
