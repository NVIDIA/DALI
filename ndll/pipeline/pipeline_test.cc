#include "ndll/pipeline/pipeline.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "ndll/common.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/batch.h"
#include "ndll/pipeline/data/buffer.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/operator.h"
#include "ndll/pipeline/operators/tjpg_decoder.h"
#include "ndll/pipeline/operators/copy_op.h"
#include "ndll/pipeline/operators/normalize_permute_op.h"
#include "ndll/pipeline/operators/resize_crop_mirror_op.h"
#include "ndll/test/ndll_main_test.h"
#include "ndll/util/image.h"

namespace ndll {

template <typename BackendPair>
class PipelineTest : public NDLLTest {
public:
  typedef typename BackendPair::TCPUBackend HostBackend;
  typedef typename BackendPair::TGPUBackend DeviceBackend;

  void SetUp() {
    NDLLTest::SetUp();
    NDLLTest::DecodeJPEGS(NDLL_RGB);
  }

  template <typename T>
  void CompareData(const T* data, const T* ground_truth, int n) {
    vector<T> tmp_cpu(n);
    CUDA_CALL(cudaMemcpy(tmp_cpu.data(), data, sizeof(T)*n, cudaMemcpyDefault));

    vector<double> abs_diff(n, 0);
    for (int i = 0; i < n; ++i) {
      abs_diff[i] = abs(double(tmp_cpu[i]) - double(ground_truth[i]));
    }
    double mean, std;
    NDLLTest::MeanStdDev(abs_diff, &mean, &std);

#ifndef NDEBUG
    cout << "num: " << abs_diff.size() << endl;
    cout << "mean: " << mean << endl;
    cout << "std: " << std << endl;
#endif
    
    ASSERT_LT(mean, 0.000001);
    ASSERT_LT(std, 0.000001);
  }
  
protected:
};

template <typename CPUBackend, typename GPUBackend, int number_of_threads>
struct PipelineTestTypes {
  typedef CPUBackend TCPUBackend;
  typedef GPUBackend TGPUBackend;
  static const int nt = number_of_threads;
};

typedef ::testing::Types<PipelineTestTypes<CPUBackend, GPUBackend, 1>,
                         PipelineTestTypes<CPUBackend, GPUBackend, 2>,
                         PipelineTestTypes<CPUBackend, GPUBackend, 3>,
                         PipelineTestTypes<CPUBackend, GPUBackend, 4>
                         > BackendTypes;
TYPED_TEST_CASE(PipelineTest, BackendTypes);

#define DECLTYPES()                                       \
  typedef typename TypeParam::TCPUBackend HostBackend;    \
  typedef typename TypeParam::TGPUBackend DeviceBackend;  \
  int num_thread = TypeParam::nt

TYPED_TEST(PipelineTest, TestSinglePrefetchOp) {
  typedef typename TypeParam::TCPUBackend HostBackend;
  int num_thread = TypeParam::nt;
  
  int batch_size = this->RandInt(1, 256);
  Pipeline pipe(batch_size, num_thread, 0, 0);

  Batch<CPUBackend> tmp_batch;
  this->MakeImageBatch(batch_size, &tmp_batch);
  
  // Create a batches of data to work with
  shared_ptr<Batch<HostBackend>> batch(new Batch<HostBackend>);
  batch->Copy(tmp_batch);

  // Add a data reader
  BatchDataReader reader(batch);
  pipe.AddDataReader(reader);

  // Add a single op to the prefetch stage
  CopyOp<HostBackend> copy_op;
  pipe.AddPrefetchOp(copy_op);

  // Build the pipeline
  pipe.Build();
  
  // Run the pipeline
  for (int i = 0; i < 5; ++i) {
    pipe.RunPrefetch();
    pipe.RunCopy();
    pipe.RunForward();

    // Verify the results
    this->CompareData(
        pipe.output_batch().template data<uint8>(),
        batch->template data<uint8>(),
        batch->size());
  }
}

TYPED_TEST(PipelineTest, TestNoOps) {
  typedef typename TypeParam::TCPUBackend HostBackend;
  int num_thread = TypeParam::nt;
  
  int batch_size = this->RandInt(1, 256);
  Pipeline pipe(batch_size, num_thread, 0, 0);

  Batch<CPUBackend> tmp_batch;
  this->MakeImageBatch(batch_size, &tmp_batch);
  
  // Create a batches of data to work with
  shared_ptr<Batch<HostBackend>> batch(new Batch<HostBackend>);
  batch->Copy(tmp_batch);

  // Add a data reader
  BatchDataReader reader(batch);
  pipe.AddDataReader(reader);

  // Build the pipeline
  pipe.Build();
  
  // Run the pipeline
  for (int i = 0; i < 5; ++i) {
    pipe.RunPrefetch();
    pipe.RunCopy();
    pipe.RunForward();

    // Verify the results
    this->CompareData(
        pipe.output_batch().template data<uint8>(),
        batch->template data<uint8>(),
        batch->size());
  }
}

TYPED_TEST(PipelineTest, TestSingleForwardOp) {
  DECLTYPES();
  int batch_size = this->RandInt(1, 256);
  Pipeline pipe(batch_size, num_thread, 0, 0);

  Batch<CPUBackend> tmp_batch;
  this->MakeImageBatch(batch_size, &tmp_batch);
  
  // Create a batches of data to work with
  shared_ptr<Batch<HostBackend>> batch(new Batch<HostBackend>);
  batch->Copy(tmp_batch);

  // Add a data reader
  BatchDataReader reader(batch);
  pipe.AddDataReader(reader);

  // Add a single op to the forward stage
  CopyOp<DeviceBackend> copy_op;
  pipe.AddForwardOp(copy_op);

  // Build the pipeline
  pipe.Build();
  
  // Run the pipeline
  for (int i = 0; i < 5; ++i) {
    pipe.RunPrefetch();
    pipe.RunCopy();
    pipe.RunForward();

    // Verify the results
    this->CompareData(
        pipe.output_batch().template data<uint8>(),
        batch->template data<uint8>(),
        batch->size());
  }
}

} // namespace ndll
