#include "ndll/pipeline/caffe2_reader_op.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(Caffe2Reader, Caffe2Reader);

OPERATOR_SCHEMA(Caffe2Reader)
  .DocStr("Read sample data from a Caffe2 LMDB")
  .NumInput(0)
  .NumOutput(32);

}  // namespace ndll

