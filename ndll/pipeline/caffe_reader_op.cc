// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/caffe_reader_op.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(CaffeReader, CaffeReader);

OPERATOR_SCHEMA(CaffeReader)
  .DocStr("Read (Image, label) pairs from a Caffe LMDB")
  .NumInput(0)
  .NumOutput(2);  // (Images, Labels)

}  // namespace ndll
