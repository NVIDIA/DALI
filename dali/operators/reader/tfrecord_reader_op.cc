// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#ifdef DALI_BUILD_PROTO3

#include <vector>
#include <string>

#include "dali/operators/reader/tfrecord_reader_op.h"

namespace dali {

namespace {

int TFRecordReaderOutputFn(const OpSpec &spec) {
  std::vector<std::string> v = spec.GetRepeatedArgument<std::string>("feature_names");
  return v.size();
}

}  // namespace

// Internal readers._tfrecord, note the triple underscore.
DALI_REGISTER_OPERATOR(readers___TFRecord, TFRecordReader, CPU);

// Common part of schema for internal readers._tfrecord and public readers.tfrecord schema.
DALI_SCHEMA(readers___TFRecordBase)
  .DocStr(R"code(Read sample data from a TensorFlow TFRecord file.)code")
  .AddArg("path",
      R"code(List of paths to TFRecord files.)code",
      DALI_STRING_VEC)
  .AddArg("index_path",
      R"code(List of paths to index files. There should be one index file for every TFRecord file.

The index files can be obtained from TFRecord files by using the ``tfrecord2idx`` script
that is distributed with DALI.)code",
      DALI_STRING_VEC);

// Internal readers._tfrecord schema.
DALI_SCHEMA(readers___TFRecord)
  .DocStr(R"code(Reads samples from a TensorFlow TFRecord file.)code")
  .OutputFn(TFRecordReaderOutputFn)
  .NumInput(0)
  .AddArg("feature_names", "Names of the features in TFRecord.",
      DALI_STRING_VEC)
  .AddArg("features", "List of features.",
      DALI_TF_FEATURE_VEC)
  .AddParent("readers___TFRecordBase")
  .AddParent("LoaderBase")
  .MakeInternal();

// Schema for the actual readers.tfrecord op exposed in Python.
// It is here for proper docstring generation. Note the double underscore.
DALI_SCHEMA(readers__TFRecord)
  .DocStr(R"code(Reads samples from a TensorFlow TFRecord file.)code")
  .AddArg("features",
      R"code(A dictionary that maps names of the TFRecord features to extract to the feature type.

Typically obtained by using the ``dali.tfrecord.FixedLenFeature`` and
``dali.tfrecord.VarLenFeature`` helper functions, which are equal to TensorFlow's
``tf.FixedLenFeature`` and ``tf.VarLenFeature`` types, respectively. For additional flexibility,
``dali.tfrecord.VarLenFeature`` supports the ``partial_shape`` parameter. If provided,
the data will be reshaped to match its value, and the first dimension will be inferred from
the data size.

If the named feature doesn't exists in the processed TFRecord entry an empty tensor is returned.
)code",
      DALI_TF_FEATURE_DICT)
  .AddParent("readers___TFRecordBase")
  .AddParent("LoaderBase");


// Deprecated alias for internal op. Necessary for deprecation warning.
DALI_REGISTER_OPERATOR(_TFRecordReader, TFRecordReader, CPU);

DALI_SCHEMA(_TFRecordReader)
    .DocStr("Legacy alias for :meth:`readers.tfrecord`.")
    .OutputFn(TFRecordReaderOutputFn)
    .NumInput(0)
    .AddParent("readers___TFRecord")
    .MakeInternal()
    .Deprecate(
        "readers__TFRecord",
        R"code(In DALI 1.0 all readers were moved into a dedicated :mod:`~nvidia.dali.fn.readers`
submodule and renamed to follow a common pattern. This is a placeholder operator with identical
functionality to allow for backward compatibility.)code");  // Deprecated in 1.0;


// Deprecated alias
DALI_SCHEMA(TFRecordReader)
    .DocStr("Legacy alias for :meth:`readers.tfrecord`.")
    .AddParent("readers__TFRecord")
    .MakeDocPartiallyHidden()
    .Deprecate(
        "readers__TFRecord",
        R"code(In DALI 1.0 all readers were moved into a dedicated :mod:`~nvidia.dali.fn.readers`
submodule and renamed to follow a common pattern. This is a placeholder operator with identical
functionality to allow for backward compatibility.)code");  // Deprecated in 1.0;

}  // namespace dali

#endif  // DALI_BUILD_PROTO3
