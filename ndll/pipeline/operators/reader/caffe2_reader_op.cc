// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/reader/caffe2_reader_op.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(Caffe2Reader, Caffe2Reader);

NDLL_OPERATOR_SCHEMA(Caffe2Reader)
  .DocStr("Read sample data from a Caffe2 LMDB")
  .NumInput(0)
  .NumOutput(1, INT_MAX);

}  // namespace ndll

