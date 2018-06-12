// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifdef NDLL_BUILD_PROTO3

#include <vector>
#include <string>

#include "ndll/pipeline/operators/reader/tfrecord_reader_op.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(_TFRecordReader, TFRecordReader, CPU);

NDLL_SCHEMA(_TFRecordReaderBase)
  .DocStr(R"code(Read sample data from a TensorFlow TFRecord file.)code")
  .AddArg("path",
      R"code(`list of string`
      List of paths to TFRecord files)code")
  .AddArg("index_path",
      R"code(`list of string`
      List of paths to index files (1 index file for every TFRecord file).
      Index files may be obtained from TFRecord files using
      `tfrecord2idx` script distributed with NDLL)code");

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
