// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifdef NDLL_BUILD_PROTO3

#include <vector>
#include <string>

#include "ndll/pipeline/operators/reader/tfrecord_reader_op.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(_TFRecordReader, TFRecordReader, CPU);

NDLL_OPERATOR_SCHEMA(_TFRecordReader)
  .OutputFn([](const OpSpec &spec) {
      std::vector<std::string> v = spec.GetRepeatedArgument<std::string>("feature_names");
      return v.size();
    })
  .NumInput(0)
  .AddArg("path", "List of paths to TFRecord files")
  .AddArg("index_path", "List of paths to index files")
  .AddArg("feature_names", "Names of the features in TFRecord")
  .AddArg("features", "List of features")
  LOADER_SCHEMA_ARGS;

}  // namespace ndll

#endif  // NDLL_BUILD_PROTO3
