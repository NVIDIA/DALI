// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/reader/caffe_reader_op.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(CaffeReader, CaffeReader, CPU);

NDLL_SCHEMA(CaffeReader)
  .DocStr("Read (Image, label) pairs from a Caffe LMDB")
  .NumInput(0)
  .NumOutput(2)  // (Images, Labels)
  .AddArg("path", 
      R"code(`string`
      Path to Caffe LMDB directory)code")
  .AddParent("LoaderBase");

}  // namespace ndll
