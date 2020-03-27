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

#include <string>

#include "dali/operators/reader/numpy_reader_op.h"

namespace dali {

DALI_REGISTER_OPERATOR(NumpyReader, NumpyReader, CPU);

DALI_SCHEMA(NumpyReader)
  .DocStr("Read Numpy arrays from a directory")
  .NumInput(0)
  .NumOutput(1)  // (Arrays)
  .AddArg("file_root",
      R"code(Path to a directory containing data files.
`NumpyReader` supports flat directory structure. `file_root` directory should contain
directories with numpy files in them.)code",
      DALI_STRING)
  .AddOptionalArg("file_filter",
      R"code(If specified, the string will be interpreted as glob string to filer 
the list of files in the sub-directories of `file_root`.)code", "")
  .AddOptionalArg("shuffle_after_epoch",
      R"code(If true, reader shuffles whole dataset after each epoch. It is exclusive with
`stick_to_shard` and `random_shuffle`.)code",
      false)
  .AddParent("LoaderBase");

}  // namespace dali
