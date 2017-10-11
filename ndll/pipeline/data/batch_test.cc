#include "ndll/pipeline/data/batch.h"

#include <gtest/gtest.h>

#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/buffer.h"
#include "ndll/pipeline/data/datum.h"

namespace ndll {

template <typename Backend>
class BatchTest : public ::testing::Test {
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

  vector<Dims> GetRandShape() {
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
protected:
  std::mt19937 rand_gen_;
};

typedef ::testing::Types<CPUBackend,
                         GPUBackend> Backends;
TYPED_TEST_CASE(BatchTest, Backends);

TYPED_TEST(BatchTest, TestResize) {
  Batch<TypeParam> batch;

  // Setup shape and offsets
  vector<Dims> shape = this->GetRandShape();
  int batch_size = shape.size();
  vector<Index> offsets;
  Index offset = 0;
  for (auto &tmp : shape) {
    offsets.push_back(offset);
    offset += Product(tmp);
  }

  // Resize the buffer
  batch.Resize(shape);
    
  // Check the internals
  ASSERT_NE(batch.template data<float>(), nullptr);
  ASSERT_EQ(batch.ndatum(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ASSERT_EQ(batch.datum_shape(i), shape[i]);
    ASSERT_EQ(batch.datum_offset(i), offsets[i]);
  }
}

TYPED_TEST(BatchTest, TestMultipleResize) {
  Batch<TypeParam> batch;

  int rand = this->RandInt(1, 20);
  vector<Dims> shape;
  vector<Index> offsets;
  int batch_size = 0;
  for (int i = 0; i < rand; ++i) {
    offsets.clear();
    // Setup shape and offsets
    shape = this->GetRandShape();
    batch_size = shape.size();
    Index offset = 0;
    for (auto &tmp : shape) {
      offsets.push_back(offset);
      offset += Product(tmp);
    }
  }

  // Resize the buffer
  batch.Resize(shape);
    
  // The only thing that should matter is the resize
  // after the call to 'data<T>()'
  ASSERT_NE(batch.template data<float>(), nullptr);
  ASSERT_EQ(batch.ndatum(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ASSERT_EQ(batch.datum_shape(i), shape[i]);
    ASSERT_EQ(batch.datum_offset(i), offsets[i]);
  }
}

TYPED_TEST(BatchTest, TestTypeChange) {
  Batch<TypeParam> batch;

  // Setup shape and offsets
  vector<Dims> shape = this->GetRandShape();
  int batch_size = shape.size();
  vector<Index> offsets;
  Index offset = 0;
  for (auto &tmp : shape) {
    offsets.push_back(offset);
    offset += Product(tmp);
  }

  // Resize the buffer
  batch.Resize(shape);
    
  // Check the internals
  ASSERT_NE(batch.template data<float>(), nullptr);
  ASSERT_EQ(batch.ndatum(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ASSERT_EQ(batch.datum_shape(i), shape[i]);
    ASSERT_EQ(batch.datum_offset(i), offsets[i]);
  }

  // Save the pointer
  void *ptr = batch.raw_data();
  size_t nbytes = batch.nbytes();
  
  // Change the data type
  batch.template data<int>();

  // Check the internals
  ASSERT_EQ(batch.ndatum(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ASSERT_EQ(batch.datum_shape(i), shape[i]);
    ASSERT_EQ(batch.datum_offset(i), offsets[i]);
  }

  // No memory allocation should have occured
  ASSERT_EQ(ptr, batch.raw_data());
  ASSERT_EQ(nbytes, batch.nbytes());
  
  // Change the data type to something smaller
  batch.template data<uint8>();
  
  // Check the internals
  ASSERT_EQ(batch.ndatum(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ASSERT_EQ(batch.datum_shape(i), shape[i]);
    ASSERT_EQ(batch.datum_offset(i), offsets[i]);
  }
  
  // No memory allocation should have occured
  ASSERT_EQ(ptr, batch.raw_data());

  // nbytes should have reduced by a factor of 4
  ASSERT_EQ(nbytes / sizeof(float) * sizeof(uint8), batch.nbytes());

  // Change the data type to something smaller
  batch.template data<double>();
  
  // Check the internals
  ASSERT_EQ(batch.ndatum(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ASSERT_EQ(batch.datum_shape(i), shape[i]);
    ASSERT_EQ(batch.datum_offset(i), offsets[i]);
  }
  
  // Size doubled, memory allocation should have occured
  ASSERT_NE(ptr, batch.raw_data());

  // nbytes should have reduced by a factor of 4
  ASSERT_EQ(nbytes / sizeof(float) * sizeof(double), batch.nbytes());
}

TYPED_TEST(BatchTest, TestShareData) {
  Batch<TypeParam> batch;

  // Setup shape and offsets
  vector<Dims> shape = this->GetRandShape();
  int batch_size = shape.size();
  vector<Index> offsets;
  Index offset = 0;
  for (auto &tmp : shape) {
    offsets.push_back(offset);
    offset += Product(tmp);
  }

  // Resize the buffer
  batch.Resize(shape);
    
  // Check the internals
  ASSERT_NE(batch.template data<float>(), nullptr);
  ASSERT_EQ(batch.ndatum(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ASSERT_EQ(batch.datum_shape(i), shape[i]);
    ASSERT_EQ(batch.datum_offset(i), offsets[i]);
  }

  // Create a new batch w/ a smaller data type
  Batch<TypeParam> batch2;
  batch2.template data<uint8>();

  // Share the data
  batch2.ShareData(batch);

  // Make sure the pointers match
  ASSERT_EQ(batch.raw_data(), batch2.raw_data());
  ASSERT_TRUE(batch2.shares_data());
  
  // Verify the default dims of the batch 2
  ASSERT_EQ(batch2.size(), batch.size() / sizeof(uint8) * sizeof(float));

  // Resize the batch2 to match the shape of batch
  batch2.Resize(shape);

  // Check the internals
  ASSERT_TRUE(batch2.shares_data());
  ASSERT_EQ(batch2.raw_data(), batch.raw_data());
  ASSERT_EQ(batch2.nbytes(), batch.nbytes() / sizeof(float) * sizeof(uint8));
  ASSERT_EQ(batch2.ndatum(), batch_size);
  ASSERT_EQ(batch2.size(), batch.size());
  for (int i = 0; i < batch_size; ++i) {
    ASSERT_EQ(batch2.datum_shape(i), shape[i]);
    ASSERT_EQ(batch2.datum_offset(i), offsets[i]);
  }

  
  // Trigger allocation through buffer API, verify we no longer share
  batch2.template data<double>();
  ASSERT_FALSE(batch2.shares_data());

  // Check the internals
  ASSERT_EQ(batch2.size(), batch.size());
  ASSERT_EQ(batch2.nbytes(), batch.nbytes() / sizeof(float) * sizeof(double));
  ASSERT_EQ(batch2.ndatum(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ASSERT_EQ(batch2.datum_shape(i), shape[i]);
    ASSERT_EQ(batch2.datum_offset(i), offsets[i]);
  }
}

} // namespace ndll
