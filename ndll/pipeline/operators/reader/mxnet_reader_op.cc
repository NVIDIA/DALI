// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/reader/mxnet_reader_op.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(MXNetReader, MXNetReader, CPU);

NDLL_OPERATOR_SCHEMA(MXNetReader)
  .DocStr("Read sample data from a MXNet RecordIO")
  .NumInput(0)
  .NumOutput(2);
}  // namespace ndll

