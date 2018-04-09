// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/reader/caffe_reader_op.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(CaffeReader, CaffeReader, CPU);

NDLL_OPERATOR_SCHEMA(CaffeReader)
  .DocStr("Read (Image, label) pairs from a Caffe LMDB")
  .NumInput(0)
  .NumOutput(2)  // (Images, Labels)
  .AddArg("path", "Path to Caffe LMDB directory")
  LOADER_SCHEMA_ARGS;

}  // namespace ndll
