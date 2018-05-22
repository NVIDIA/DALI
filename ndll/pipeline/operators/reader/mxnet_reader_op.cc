// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/reader/mxnet_reader_op.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(MXNetReader, MXNetReader, CPU);

NDLL_SCHEMA(MXNetReader)
  .DocStr("Read sample data from a MXNet RecordIO")
  .NumInput(0)
  .NumOutput(2)
  .AddArg("path", "List of paths to RecordIO files")
  .AddArg("index_path", "List of paths to index files")
  LOADER_SCHEMA_ARGS;

}  // namespace ndll

