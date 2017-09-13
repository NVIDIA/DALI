#include "ndll/pipeline/data/batch.h"

#include <gtest/gtest.h>

#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/buffer.h"

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
                         PinnedCPUBackend,
                         GPUBackend> Backends;
TYPED_TEST_CASE(BatchTest, Backends);

TYPED_TEST(BatchTest, TestResize) {
  try {
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
  } catch (NDLLException &e) {
    FAIL() << e.what();
  }
}

TYPED_TEST(BatchTest, TestMultipleResize) {
  try {
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
  } catch (NDLLException &e) {
    FAIL() << e.what();
  }
}

TYPED_TEST(BatchTest, TestGetDatum) {
  try {
    // Setup batch of samples
    Batch<TypeParam> batch;
    vector<Dims> shape = this->GetRandShape();
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
  } catch (NDLLException &e) {
    FAIL() << e.what();
  }
}

} // namespace ndll
