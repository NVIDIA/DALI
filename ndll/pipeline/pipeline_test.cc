#include "ndll/pipeline/pipeline.h"

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
#include "ndll/util/image.h"

namespace ndll {

template <typename BackendPair>
class PipelineTest : public NDLLTest {
public:
  typedef typename BackendPair::TCPUBackend HostBackend;
  typedef typename BackendPair::TGPUBackend DeviceBackend;
  
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
  
// TODO(tgale): This isnt actually a test. Need to
// build real tests for the pipeline
TYPED_TEST(PipelineTest, DISABLED_TestBuildPipeline) {
  DECLTYPES();
  try {
    int batch_size = 1;
    // Create the pipeline
    Pipeline<HostBackend, DeviceBackend> pipe(batch_size, 1, 0, 8, true, 0);
    
    // Add a decoder and some transformers
    TJPGDecoder<HostBackend> jpg_decoder(true);
    pipe.AddDecoder(jpg_decoder);

#ifndef NDEBUG
    // Add a dump image op
    DumpImageOp<HostBackend> dump_image_op;
    pipe.AddPrefetchOp(dump_image_op);
#endif
    
    // Add a resize+crop+mirror op
    ResizeCropMirrorOp<HostBackend> resize_crop_mirror_op(
        true, false, 256, 480, true, 224, 224, 0.5f);
    pipe.AddPrefetchOp(resize_crop_mirror_op);
    
#ifndef NDEBUG
    // Add a dump image op
    DumpImageOp<HostBackend> dump_image_op2;
    pipe.AddPrefetchOp(dump_image_op2);
#endif
    
    // Add normalize permute op
    NormalizePermuteOp<DeviceBackend, float> norm_permute_op(
        {128, 128, 128}, {1, 1, 1}, 224, 224, 3);
    pipe.AddForwardOp(norm_permute_op);
    
    Batch<HostBackend> *batch = CreateJPEGBatch<HostBackend>(
        this->jpegs_, this->jpeg_sizes_, batch_size);
    Batch<DeviceBackend> output_batch;
    
    // Build and run the pipeline
    pipe.Build(batch->type());

    pipe.Print();
    
    pipe.RunPrefetch(batch);
    pipe.RunCopy();
    pipe.RunForward(&output_batch);

#ifndef NDEBUG
    DumpCHWImageBatchToFile<float>(output_batch);
#endif
    
    CUDA_CALL(cudaDeviceSynchronize());
    delete batch;
  } catch (std::runtime_error &e) {
    FAIL() << e.what();
  }
}

} // namespace ndll
