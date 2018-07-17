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

#include "dali/pipeline/operators/reader/file_reader_op.h"

namespace dali {

DALI_REGISTER_OPERATOR(FileReader, FileReader, CPU);

DALI_SCHEMA(FileReader)
  .DocStr("Read (Image, label) pairs from a directory")
  .NumInput(0)
  .NumOutput(2)  // (Images, Labels)
  .AddArg("file_root",
      R"code(Path to a directory containing data files.)code",
      DALI_STRING)
  .AddOptionalArg("file_list",
      R"code(Path to the file with a list of pairs ``file label``
(leave empty to traverse the `file_root` directory to obtain files and labels))code",
      std::string())
  .AddParent("LoaderBase");

}  // namespace dali

