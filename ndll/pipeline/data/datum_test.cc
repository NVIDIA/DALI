#include "ndll/pipeline/data/datum.h"

#include <gtest/gtest.h>

#include "ndll/pipeline/data/batch.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/buffer.h"
#include "ndll/test/ndll_main_test.h"

namespace ndll {

template <typename Backend>
class DatumTest : public NDLLTest {
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
TYPED_TEST_CASE(DatumTest, Backends);

TYPED_TEST(DatumTest, TestResize) {
  Datum<TypeParam> datum;

  // Get shape
  vector<Index> shape = this->GetRandShape();
  datum.Resize(shape);

  // Verify the settings
  ASSERT_NE(datum.template data<float>(), nullptr);
  ASSERT_EQ(datum.size(), Product(shape));
  ASSERT_TRUE(datum.owned());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(datum.shape()[i], shape[i]);
  }
}

TYPED_TEST(DatumTest, TestMultipleResize) {
  Datum<TypeParam> datum;

  int num = this->RandInt(2, 20);
  for (int i = 0; i < num; ++i) {
    // Get shape
    vector<Index> shape = this->GetRandShape();
    datum.Resize(shape);
      
    // Verify the settings
    ASSERT_NE(datum.template data<float>(), nullptr);
    ASSERT_EQ(datum.size(), Product(shape));
    ASSERT_TRUE(datum.owned());
    for (size_t i = 0; i < shape.size(); ++i) {
      ASSERT_EQ(datum.shape()[i], shape[i]);      
    }
  }
}

TYPED_TEST(DatumTest, TestGetDatum) {
  // Setup batch of samples
  Batch<TypeParam> batch;
  vector<Dims> shape = this->GetRandBatchShape();
  int batch_size = shape.size();
  batch.Resize(shape);
  batch.template data<float>();

  // Wrap a sample
  int datum_idx = this->RandInt(0, batch_size-1);
  Datum<TypeParam> datum(&batch, datum_idx);

  // Check the dims, size, etc.
  ASSERT_EQ(datum.size(), Product(batch.datum_shape(datum_idx)));
  ASSERT_EQ(datum.type(), batch.type());
  ASSERT_EQ(datum.shape(), batch.datum_shape(datum_idx));
  ASSERT_EQ(datum.nbytes(),
      Product(batch.datum_shape(datum_idx))*batch.type().size());
  ASSERT_EQ(datum.template data<float>(), batch.template datum<float>(datum_idx));
  ASSERT_EQ(datum.owned(), false);

  // Now resize the data
  vector<Index> new_shape = this->GetRandShape();
  datum.Resize(new_shape);

  // Verify the settings
  ASSERT_NE(datum.template data<float>(), nullptr);
  ASSERT_EQ(datum.size(), Product(new_shape));
  ASSERT_TRUE(datum.owned());
  for (size_t i = 0; i < new_shape.size(); ++i) {
    ASSERT_EQ(datum.shape()[i], new_shape[i]);
  }
}

TYPED_TEST(DatumTest, TestTypeChange) {
  Datum<TypeParam> datum;

  // Get shape
  vector<Index> shape = this->GetRandShape();
  datum.Resize(shape);

  // Verify the settings
  ASSERT_NE(datum.template data<float>(), nullptr);
  ASSERT_EQ(datum.size(), Product(shape));
  ASSERT_TRUE(datum.owned());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(datum.shape()[i], shape[i]);
  }

  // Save the pointer
  void *ptr = datum.raw_data();
  size_t nbytes = datum.nbytes();

  // Change the data type
  datum.template data<int>();

  // Verify the settings
  ASSERT_EQ(datum.size(), Product(shape));
  ASSERT_TRUE(datum.owned());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(datum.shape()[i], shape[i]);
  }

  // No allocation should have occured
  ASSERT_EQ(ptr, datum.raw_data());
  ASSERT_EQ(nbytes, datum.nbytes());

  // Change the data type to something smaller
  datum.template data<uint8>();

  // Verify the settings
  ASSERT_EQ(datum.size(), Product(shape));
  ASSERT_TRUE(datum.owned());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(datum.shape()[i], shape[i]);
  }

  // No allocation should have occured
  ASSERT_EQ(ptr, datum.raw_data());
  ASSERT_EQ(nbytes / sizeof(float) * sizeof(uint8), datum.nbytes());

  // Change the data type to something bigger
  datum.template data<double>();

  // Verify the settings
  ASSERT_EQ(datum.size(), Product(shape));
  ASSERT_TRUE(datum.owned());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(datum.shape()[i], shape[i]);
  }

  // Allocation should have occured
  ASSERT_NE(ptr, datum.raw_data());
  ASSERT_EQ(nbytes / sizeof(float) * sizeof(double), datum.nbytes());
}

} // namespace ndll
