// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifdef NDLL_BUILD_PROTO3

#include <vector>
#include <string>

#include "ndll/pipeline/operators/reader/tfrecord_reader_op.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(_TFRecordReader, TFRecordReader, CPU);

NDLL_SCHEMA(_TFRecordReaderBase)
  .AddArg("path", "List of paths to TFRecord files")
  .AddArg("index_path", "List of paths to index files");

NDLL_SCHEMA(_TFRecordReader)
  .OutputFn([](const OpSpec &spec) {
      std::vector<std::string> v = spec.GetRepeatedArgument<std::string>("feature_names");
      return v.size();
    })
  .NumInput(0)
  .AddArg("feature_names", "Names of the features in TFRecord")
  .AddArg("features", "List of features")
  .AddParent("_TFRecordReaderBase")
  .AddParent("LoaderBase");

// Schema for the actual TFRecordReader op exposed
// in Python. It is here for proper docstring generation
NDLL_SCHEMA(TFRecordReader)
  .AddArg("features",
      R"code(`dict of (string, ndll.tfrecord.Feature)`
      Dictionary of names and configuration of features existing in TFRecord file.
      Typically obtained using helper functions `ndll.tfrecord.FixedLenFeature`
      and `ndll.tfrecord.VarLenFeature`, they are equivalent to TensorFlow's `tf.FixedLenFeature` and
      `tf.VarLenFeature` respectively)code")
  .AddParent("_TFRecordReaderBase")
  .AddParent("LoaderBase");

}  // namespace ndll

#endif  // NDLL_BUILD_PROTO3
