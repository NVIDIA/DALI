#include "ndll/pipeline/pipeline.h"

#include <gtest/gtest.h>

#include "ndll/common.h"
#include "ndll/pipeline/backend.h"
#include "ndll/pipeline/buffer.h"
#include "ndll/pipeline/operator.h"

namespace ndll {

class PipelineTest : public ::testing::Test {
public:
protected:
};

TEST_F(PipelineTest, TestBuildPipeline) {
  try {
  Pipeline<CPUBackend, GPUBackend> pipe(4, 0, 8, true);

  Decoder<CPUBackend> dec;
  pipe.AddPrefetchOp(dec);
  pipe.Build();

  int batch_size = 32, sample_dim = 128;
  Buffer<CPUBackend> input;
  input.Resize({batch_size, sample_dim});

  // Run the pipeline
  pipe.RunPrefetch(&input);
  
  } catch (NDLLException &e) {
    FAIL() << e.what();
  }
}

} // namespace ndll
