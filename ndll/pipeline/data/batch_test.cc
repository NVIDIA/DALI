#include "ndll/pipeline/data/batch.h"

#include <gtest/gtest.h>

#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/buffer.h"
#include "ndll/pipeline/data/sample.h"
#include "ndll/test/ndll_test.h"

namespace ndll {

template <typename Backend>
class BatchTest : public NDLLTest {
public:
  vector<Dims> GetRandShape() {
    int batch_size = this->RandInt(1, 128);
    vector<Dims> shape(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      int dims = this->RandInt(1, 3);
      vector<Index> sample_shape(dims, 0);
      for (int j = 0; j < dims; ++j) {
        sample_shape[j] = this->RandInt(1, 512);
      }
      shape[i] = sample_shape;
    }
    return shape;
  }

  vector<Dims> GetSmallRandShape() {
    int batch_size = this->RandInt(1, 32);
    vector<Dims> shape(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      int dims = this->RandInt(1, 3);
      vector<Index> sample_shape(dims, 0);
      for (int j = 0; j < dims; ++j) {
        sample_shape[j] = this->RandInt(1, 64);
      }
      shape[i] = sample_shape;
    }
    return shape;
  }
  
protected:
};

typedef ::testing::Types<CPUBackend,
                         GPUBackend> Backends;
TYPED_TEST_CASE(BatchTest, Backends);

class DummyType {
public:
  DummyType(int size = 2) : ptr_(nullptr), size_(size) {
    ptr_ = new float[size];
  }

  ~DummyType() {
    delete[] ptr_;
  }

  float *ptr_;
  int size_;
};

class DummyType2 {
public:
  DummyType2(int size = 1) : ptr_(nullptr), size_(size), id_(5) {
    ptr_ = new float[size];
  }

  ~DummyType2() {
    delete[] ptr_;
  }

  float *ptr_;
  int size_;
  int id_;
};

NDLL_REGISTER_TYPE(DummyType);
NDLL_REGISTER_TYPE(DummyType2);

TYPED_TEST(BatchTest, TestDataTypeConstructor) {
  if (std::is_same<TypeParam, GPUBackend>::value) return;
  Batch<TypeParam> batch;

  // Setup shape & resize the buffer
  vector<Dims> shape = this->GetSmallRandShape();
  batch.Resize(shape);

  batch.template mutable_data<DummyType>();
  
  for (int i = 0; i < batch.size(); ++i) {
    // verify that the internal data has been constructed
    DummyType &obj = batch.template mutable_data<DummyType>()[i];
    ASSERT_EQ(obj.size_, 2);
    ASSERT_NE(obj.ptr_, nullptr);
  }

  // Switch the type to DummyType2
  batch.template mutable_data<DummyType2>();
  for (int i = 0; i < batch.size(); ++i) {
    // verify that the internal data has been constructed
    DummyType2 &obj = batch.template mutable_data<DummyType2>()[i];
    ASSERT_EQ(obj.size_, 1);
    ASSERT_EQ(obj.id_, 5);
    ASSERT_NE(obj.ptr_, nullptr);
  }
}

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
  ASSERT_NE(batch.template mutable_data<float>(), nullptr);
  ASSERT_EQ(batch.nsample(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ASSERT_EQ(batch.sample_shape(i), shape[i]);
    ASSERT_EQ(batch.sample_offset(i), offsets[i]);
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
  // after the call to 'mutable_data<T>()'
  ASSERT_NE(batch.template mutable_data<float>(), nullptr);
  ASSERT_EQ(batch.nsample(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ASSERT_EQ(batch.sample_shape(i), shape[i]);
    ASSERT_EQ(batch.sample_offset(i), offsets[i]);
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
  ASSERT_NE(batch.template mutable_data<float>(), nullptr);
  ASSERT_EQ(batch.nsample(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ASSERT_EQ(batch.sample_shape(i), shape[i]);
    ASSERT_EQ(batch.sample_offset(i), offsets[i]);
  }

  // Save the pointer
  const void *ptr = batch.raw_data();
  size_t nbytes = batch.nbytes();
  
  // Change the data type
  batch.template mutable_data<int>();

  // Check the internals
  ASSERT_EQ(batch.nsample(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ASSERT_EQ(batch.sample_shape(i), shape[i]);
    ASSERT_EQ(batch.sample_offset(i), offsets[i]);
  }

  // No memory allocation should have occured
  ASSERT_EQ(ptr, batch.raw_data());
  ASSERT_EQ(nbytes, batch.nbytes());
  
  // Change the data type to something smaller
  batch.template mutable_data<uint8>();
  
  // Check the internals
  ASSERT_EQ(batch.nsample(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ASSERT_EQ(batch.sample_shape(i), shape[i]);
    ASSERT_EQ(batch.sample_offset(i), offsets[i]);
  }
  
  // No memory allocation should have occured
  ASSERT_EQ(ptr, batch.raw_data());

  // nbytes should have reduced by a factor of 4
  ASSERT_EQ(nbytes / sizeof(float) * sizeof(uint8), batch.nbytes());

  // Change the data type to something smaller
  batch.template mutable_data<double>();
  
  // Check the internals
  ASSERT_EQ(batch.nsample(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ASSERT_EQ(batch.sample_shape(i), shape[i]);
    ASSERT_EQ(batch.sample_offset(i), offsets[i]);
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
  ASSERT_NE(batch.template mutable_data<float>(), nullptr);
  ASSERT_EQ(batch.nsample(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ASSERT_EQ(batch.sample_shape(i), shape[i]);
    ASSERT_EQ(batch.sample_offset(i), offsets[i]);
  }

  // Create a new batch w/ a smaller data type
  Batch<TypeParam> batch2;
  batch2.template mutable_data<uint8>();

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
  ASSERT_EQ(batch2.nsample(), batch_size);
  ASSERT_EQ(batch2.size(), batch.size());
  for (int i = 0; i < batch_size; ++i) {
    ASSERT_EQ(batch2.sample_shape(i), shape[i]);
    ASSERT_EQ(batch2.sample_offset(i), offsets[i]);
  }

  
  // Trigger allocation through buffer API, verify we no longer share
  batch2.template mutable_data<double>();
  ASSERT_FALSE(batch2.shares_data());

  // Check the internals
  ASSERT_EQ(batch2.size(), batch.size());
  ASSERT_EQ(batch2.nbytes(), batch.nbytes() / sizeof(float) * sizeof(double));
  ASSERT_EQ(batch2.nsample(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ASSERT_EQ(batch2.sample_shape(i), shape[i]);
    ASSERT_EQ(batch2.sample_offset(i), offsets[i]);
  }
}

} // namespace ndll
