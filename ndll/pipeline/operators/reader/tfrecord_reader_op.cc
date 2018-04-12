// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifdef NDLL_BUILD_PROTO3

#include "ndll/pipeline/operators/reader/tfrecord_reader_op.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(_TFRecordReader, TFRecordReader, CPU);

NDLL_OPERATOR_SCHEMA(_TFRecordReader)
  .OutputFn([](const OpSpec &spec) {
      return spec.NumOutput();
    })
  // TODO(ptredak): check if MaxNumOutputs is used for anything important
  // and if not, remove this limit
  .NumOutput(1, 10)
  .NumInput(0)
  .AddArg("path", "List of paths to TFRecord files")
  .AddArg("index_path", "List of paths to index files")
  .AddArg("feature_names", "Names of the features in TFRecord")
  .AddArg("features", "List of features")
  LOADER_SCHEMA_ARGS;

}  // namespace ndll

#endif  // NDLL_BUILD_PROTO3
