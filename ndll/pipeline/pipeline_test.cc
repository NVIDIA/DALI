#include "ndll/pipeline/pipeline.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "ndll/common.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/batch.h"
#include "ndll/pipeline/data/buffer.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/operator.h"
// #include "ndll/pipeline/operators/tjpg_decoder.h"
#include "ndll/pipeline/operators/copy_op.h"
// #include "ndll/pipeline/operators/normalize_permute_op.h"
// #include "ndll/pipeline/operators/resize_crop_mirror_op.h"
#include "ndll/test/ndll_test.h"
#include "ndll/util/image.h"

namespace ndll {

template <typename ThreadCount>
class PipelineTest : public NDLLTest {
public:
  void SetUp() {
    NDLLTest::SetUp();
    NDLLTest::DecodeJPEGS(NDLL_RGB);
  }

  template <typename T>
  void CompareData(const T* data, const T* ground_truth, int n) {
    CUDA_CALL(cudaDeviceSynchronize());
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

template <int number_of_threads>
struct ThreadCount {
  static const int nt = number_of_threads;
};

typedef ::testing::Types<ThreadCount<1>,
                         ThreadCount<2>,
                         ThreadCount<3>,
                         ThreadCount<4>> NumThreads;
TYPED_TEST_CASE(PipelineTest, NumThreads);

/*
TYPED_TEST(PipelineTest, TestSinglePrefetchOp) {
  int num_thread = TypeParam::nt;
  int batch_size = this->jpegs_.size();
  cudaStream_t stream;
  CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  
  Pipeline pipe(batch_size, num_thread, stream, 0, true);

  Batch<CPUBackend> *batch =
    CreateJPEGBatch<CPUBackend>(this->jpegs_, this->jpeg_sizes_, batch_size);
  
  // Add a data reader
  pipe.AddDataReader(
      OpSpec("BatchDataReader")
      .AddArg("jpeg_folder", image_folder)
      );

  // Add a single prefetch op
  pipe.AddTransform(OpSpec("CopyOp")
      .AddArg("stage", "Prefetch"));

  // Build the pipeline
  pipe.Build();
  
  // Run the pipeline
  for (int i = 0; i < 5; ++i) {
    pipe.RunPrefetch();
    pipe.RunCopy();
    pipe.RunForward();

    CUDA_CALL(cudaStreamSynchronize(stream));
        
    // Verify the results
    this->CompareData(
        pipe.output_batch().template data<uint8>(),
        batch->template data<uint8>(),
        batch->size());
  }
}

TYPED_TEST(PipelineTest, TestNoOps) {
  int num_thread = TypeParam::nt;
  
  int batch_size = this->jpegs_.size();
  Pipeline pipe(batch_size, num_thread, 0, 0, true);

  Batch<CPUBackend> *batch =
    CreateJPEGBatch<CPUBackend>(this->jpegs_, this->jpeg_sizes_, batch_size);
    
  
  // Add a data reader
  pipe.AddDataReader(
      OpSpec("BatchDataReader")
      .AddArg("jpeg_folder", image_folder)
      );
  
  // Build the pipeline
  pipe.Build();
  
  // Run the pipeline
  for (int i = 0; i < 5; ++i) {
    pipe.RunPrefetch();
    pipe.RunCopy();
    pipe.RunForward();

    CUDA_CALL(cudaStreamSynchronize(0));
    
    // Verify the results
    this->CompareData(
        pipe.output_batch().template data<uint8>(),
        batch->template data<uint8>(),
        batch->size());
  }
}

TYPED_TEST(PipelineTest, TestSingleForwardOp) {
  int num_thread = TypeParam::nt;

  int batch_size = this->jpegs_.size();
  Pipeline pipe(batch_size, num_thread, 0, 0, true);

  Batch<CPUBackend> *batch =
    CreateJPEGBatch<CPUBackend>(this->jpegs_, this->jpeg_sizes_, batch_size);
  
  // Add a data reader
  pipe.AddDataReader(
      OpSpec("BatchDataReader")
      .AddArg("jpeg_folder", image_folder)
      );
  
  // Add a single op to the forward stage
  pipe.AddTransform(OpSpec("CopyOp")
      .AddArg("stage", "Forward")
      );

  // Build the pipeline
  pipe.Build();
  
  // Run the pipeline
  for (int i = 0; i < 5; ++i) {
    pipe.RunPrefetch();
    pipe.RunCopy();
    pipe.RunForward();

    // Verify the results
    CUDA_CALL(cudaStreamSynchronize(0));
    this->CompareData(
        pipe.output_batch().template data<uint8>(),
        batch->template data<uint8>(),
        batch->size());
  }
}
*/
} // namespace ndll
