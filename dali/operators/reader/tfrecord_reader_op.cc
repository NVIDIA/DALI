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

DALI_REGISTER_OPERATOR(_TFRecordReader, TFRecordReader, CPU);

DALI_SCHEMA(_TFRecordReaderBase)
  .DocStr(R"code(Read sample data from a TensorFlow TFRecord file.)code")
  .AddArg("path",
      R"code(List of paths to TFRecord files.)code",
      DALI_STRING_VEC)
  .AddArg("index_path",
      R"code(List of paths to index files (1 index file for every TFRecord file).
Index files may be obtained from TFRecord files using
`tfrecord2idx` script distributed with DALI.)code",
      DALI_STRING_VEC);

DALI_SCHEMA(_TFRecordReader)
  .DocStr(R"code(Read sample data from a TensorFlow TFRecord file.)code")
  .OutputFn([](const OpSpec &spec) {
      std::vector<std::string> v = spec.GetRepeatedArgument<std::string>("feature_names");
      return v.size();
    })
  .NumInput(0)
  .AddArg("feature_names", "Names of the features in TFRecord.",
      DALI_STRING_VEC)
  .AddArg("features", "List of features.",
      DALI_TF_FEATURE_VEC)
  .AddParent("_TFRecordReaderBase")
  .AddParent("LoaderBase")
  .MakeInternal();

// Schema for the actual TFRecordReader op exposed
// in Python. It is here for proper docstring generation
DALI_SCHEMA(TFRecordReader)
  .DocStr(R"code(Read sample data from a TensorFlow TFRecord file.)code")
  .AddArg("features",
      R"code(Dictionary of names and configuration of features existing in TFRecord file.
Typically obtained using helper functions `dali.tfrecord.FixedLenFeature`
and `dali.tfrecord.VarLenFeature`, they are equivalent to TensorFlow's `tf.FixedLenFeature` and
`tf.VarLenFeature` respectively. For more flexibility `dali.tfrecord.VarLenFeature`
supports `partial_shape` parameter. If provided, data will be reshaped to match its value.
First dimension will be inferred from the data size.)code",
      DALI_TF_FEATURE_DICT)
  .AddParent("_TFRecordReaderBase")
  .AddParent("LoaderBase");

}  // namespace dali

#endif  // DALI_BUILD_PROTO3
