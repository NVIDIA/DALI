// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include <string>

#include "ndll/pipeline/operators/reader/file_reader_op.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(FileReader, FileReader, CPU);

NDLL_SCHEMA(FileReader)
  .DocStr("Read (Image, label) pairs from a directory")
  .NumInput(0)
  .NumOutput(2)  // (Images, Labels)
  .AddArg("file_root", 
      R"code(`string`
      Path to a directory containing data files)code")
  .AddOptionalArg("file_list",
      R"code(`string`
      Path to the file with a list of pairs ``file label``
      (leave empty to traverse the `file_root` directory to obtain files and labels))code",
      std::string())
  .AddParent("LoaderBase");

}  // namespace ndll

