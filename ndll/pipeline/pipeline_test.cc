#include "ndll/pipeline/pipeline.h"

#include <gtest/gtest.h>

#include "ndll/common.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/batch.h"
#include "ndll/pipeline/data/buffer.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/operators/operator.h"

namespace ndll {

template <typename BackendPair>
class PipelineTest : public ::testing::Test {
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
    int batch_size = this->RandInt(0, 128);
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

template <typename CPUBackend, typename GPUBackend>
struct PipelineTestTypes {
  typedef CPUBackend TCPUBackend;
  typedef GPUBackend TGPUBackend;
};

typedef ::testing::Types<PipelineTestTypes<CPUBackend, GPUBackend>,
                         PipelineTestTypes<PinnedCPUBackend, GPUBackend>
                         > BackendTypes;
TYPED_TEST_CASE(PipelineTest, BackendTypes);

#define DECLTYPES()                                       \
  typedef typename TypeParam::TCPUBackend HostBackend;    \
  typedef typename TypeParam::TGPUBackend DeviceBackend
  
TYPED_TEST(PipelineTest, TestBuildPipeline) {
  DECLTYPES();
  try {
    // Create the pipeline
    Pipeline<HostBackend, DeviceBackend> pipe(4, 0, 8, true);

    // Add a decoder and some transformers
  } catch (NDLLException &e) {
    FAIL() << e.what();
  }
}

} // namespace ndll
