// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/reader/file_reader_op.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(FileReader, FileReader);

NDLL_OPERATOR_SCHEMA(FileReader)
  .DocStr("Read individual (Image, label) pairs from a list")
  .NumInput(0)
  .NumOutput(2);  // (Images, Labels)

}  // namespace ndll

