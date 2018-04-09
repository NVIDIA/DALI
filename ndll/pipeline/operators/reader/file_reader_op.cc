// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/reader/file_reader_op.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(FileReader, FileReader, CPU);

NDLL_OPERATOR_SCHEMA(FileReader)
  .DocStr("Read individual (Image, label) pairs from a list")
  .NumInput(0)
  .NumOutput(2)  // (Images, Labels)
  .AddArg("file_root", "Path to directory containing data files")
  .AddOptionalArg("file_list", "Path to the file with a list of pairs \"file label\"")
  LOADER_SCHEMA_ARGS;

}  // namespace ndll

