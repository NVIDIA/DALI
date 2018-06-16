// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifdef DALI_BUILD_PROTO3

#include <vector>
#include <string>

#include "dali/pipeline/operators/reader/tfrecord_reader_op.h"

namespace dali {

DALI_REGISTER_OPERATOR(_TFRecordReader, TFRecordReader, CPU);

DALI_SCHEMA(_TFRecordReaderBase)
  .DocStr(R"code(Read sample data from a TensorFlow TFRecord file.)code")
  .AddArg("path",
      R"code(`list of string`
      List of paths to TFRecord files)code")
  .AddArg("index_path",
      R"code(`list of string`
      List of paths to index files (1 index file for every TFRecord file).
      Index files may be obtained from TFRecord files using
      `tfrecord2idx` script distributed with DALI)code");

DALI_SCHEMA(_TFRecordReader)
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
DALI_SCHEMA(TFRecordReader)
  .AddArg("features",
      R"code(`dict of (string, dali.tfrecord.Feature)`
      Dictionary of names and configuration of features existing in TFRecord file.
      Typically obtained using helper functions `dali.tfrecord.FixedLenFeature`
      and `dali.tfrecord.VarLenFeature`, they are equivalent to TensorFlow's `tf.FixedLenFeature` and
      `tf.VarLenFeature` respectively)code")
  .AddParent("_TFRecordReaderBase")
  .AddParent("LoaderBase");

}  // namespace dali

#endif  // DALI_BUILD_PROTO3
