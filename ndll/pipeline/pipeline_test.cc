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
#include "ndll/pipeline/operators/normalize_permute_op.h"
#include "ndll/pipeline/operators/resize_crop_mirror_op.h"
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
      CUDA_CALL(cudaMemcpy(batch->raw_datum(i),
              jpegs_[i], jpeg_sizes_[i],
              cudaMemcpyDefault));
    }
    return batch;
  }

  template <typename T, typename Backend>
  void DumpHWCImageBatchToFile(Batch<Backend> &batch) {
    int batch_size = batch.ndatum();
    for (int i = 0; i < batch_size; ++i) {
      vector<Index> shape = batch.datum_shape(i);
      assert(shape.size() == 3);
      int h = shape[0], w = shape[1], c = shape[2];

      this->DumpToFile((T*)batch.raw_datum(i), h, w, c, w*c, std::to_string(i) + "-batch");
    }
    
  }

    template <typename T, typename Backend>
  void DumpCHWImageBatchToFile(Batch<Backend> &batch) {
    int batch_size = batch.ndatum();
    for (int i = 0; i < batch_size; ++i) {
      vector<Index> shape = batch.datum_shape(i);
      assert(shape.size() == 3);
      int c = shape[0], h = shape[1], w = shape[2];

      this->DumpCHWToFile((T*)batch.raw_datum(i), h, w, c, std::to_string(i) + "-batch");
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
    int batch_size = 1;
    // Create the pipeline
    Pipeline<HostBackend, DeviceBackend> pipe(batch_size, 1, 0, 8, true);
    
    // Add a decoder and some transformers
    TJPGDecoder<HostBackend> jpg_decoder(true);
    pipe.AddDecoder(jpg_decoder);

    // Add a dump image op
    DumpImageOp<HostBackend> dump_image_op;
    pipe.AddPrefetchOp(dump_image_op);

    // Add a resize+crop+mirror op
    ResizeCropMirrorOp<HostBackend> resize_crop_mirror_op(
        true, false, 256, 480, true, 224, 224, 0.5f);
    pipe.AddPrefetchOp(resize_crop_mirror_op);

    // Add a dump image op
    DumpImageOp<HostBackend> dump_image_op2;
    pipe.AddPrefetchOp(dump_image_op2);

    // Add normalize permute op
    NormalizePermuteOp<DeviceBackend, float> norm_permute_op(
        {128, 128, 128}, {1, 1, 1}, 224, 224, 3);
    pipe.AddForwardOp(norm_permute_op);
    
    Batch<HostBackend> *batch = this->template CreateJPEGBatch<HostBackend>(batch_size);
    Batch<DeviceBackend> output_batch;
    
    // Build and run the pipeline
    pipe.Build(batch->type());

    pipe.Print();
    
    pipe.RunPrefetch(batch);
    pipe.RunCopy();
    pipe.RunForward(&output_batch);

    this->template DumpCHWImageBatchToFile<float>(output_batch);

    CUDA_CALL(cudaDeviceSynchronize());
    delete batch;
  } catch (std::runtime_error &e) {
    FAIL() << e.what();
  }
}

} // namespace ndll
