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
  Pipeline<CPUBackend, GPUBackend> pipe(6, 4, 0, 8, true);

  Buffer<GPUBackend> buf, buf2;
  Decoder<GPUBackend> dec;
  dec.Run(buf, &buf2);

  Buffer<CPUBackend> cbuf, cbuf2;
  Decoder<CPUBackend> dec2;
  dec2.Run(cbuf, &cbuf2, 0);
  
  // for (int i = 0; i < 100; ++i) {
  //   Operator op;
  //   cout << op.id() << endl;
  //   pipe.AddPrefetchOp(op);
  //   cout << op.id() << endl;
  // }
}

} // namespace ndll
