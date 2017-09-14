#include "ndll/pipeline/pipeline.h"

#include <cassert>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "ndll/common.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/batch.h"
#include "ndll/pipeline/data/buffer.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/operators/operator.h"
#include "ndll/pipeline/operators/tjpg_decoder.h"
#include "ndll/pipeline/operators/copy_op.h"
#include "ndll/test/ndll_main_test.h"

namespace ndll {

template <typename BackendPair>
class PipelineTest : public NDLLTest {
public:
  typedef typename BackendPair::TCPUBackend HostBackend;
  typedef typename BackendPair::TGPUBackend DeviceBackend;

  template <typename Backend>
  auto CreateJPEGBatch(int size) -> Batch<Backend>* {
    assert(size_t(size) < jpegs_.size());
    Batch<Backend> *batch = new Batch<Backend>();
    // Create the shape
    vector<Dims> shape(size);
    for (int i = 0; i < size; ++i) {
      shape[i] = {Index(jpeg_sizes_[i])};
    }
    batch->Resize(shape);

    // Copy in the data
    batch->template data<uint8>();
    for (int i = 0; i < size; ++i) {
      TEST_CUDA(cudaMemcpy(batch->raw_datum(i),
              jpegs_[i], jpeg_sizes_[i],
              cudaMemcpyDefault));
    }
    return batch;
  }

  template <typename T, typename Backend>
  void DumpImageBatchToFile(Batch<Backend> &batch) {
    int batch_size = batch.ndatum();
    for (int i = 0; i < batch_size; ++i) {
      vector<Index> shape = batch.datum_shape(i);
      assert(shape.size() == 3);
      int h = shape[0], w = shape[1], c = shape[2];

      this->DumpToFile((T*)batch.raw_datum(i), h, w, c, w*c, std::to_string(i) + "-batch");
    }
    
  }
  
protected:
  
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
    Pipeline<HostBackend, DeviceBackend> pipe(1, 0, 8, true);
    
    // Add a decoder and some transformers
    TJPGDecoder<HostBackend> jpg_decoder(pipe.num_thread(), pipe.stream_pool(), true);
    pipe.AddDecoder(jpg_decoder);

    // Add a dump image op
    DumpImageOp<HostBackend> dump_image_op(pipe.num_thread(), pipe.stream_pool());
    pipe.AddPrefetchOp(dump_image_op);
    
    Batch<HostBackend> *batch = this->template CreateJPEGBatch<HostBackend>(4);
    Batch<DeviceBackend> output_batch;
    TypeMeta input_type = batch->type();

    // Build and run the pipeline
    pipe.Build(&input_type);

    pipe.Print();
    
    pipe.RunPrefetch(batch);
    pipe.RunCopy();
    pipe.RunForward(&output_batch);

    this->template DumpImageBatchToFile<uint8>(output_batch);

    TEST_CUDA(cudaDeviceSynchronize());
    delete batch;
  } catch (NDLLException &e) {
    FAIL() << e.what();
  }
}

} // namespace ndll
