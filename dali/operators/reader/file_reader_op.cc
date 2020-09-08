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

#include "dali/operators/reader/file_reader_op.h"

namespace dali {

DALI_REGISTER_OPERATOR(FileReader, FileReader, CPU);

DALI_SCHEMA(FileReader)
  .DocStr("Reads (file, label) pairs from a directory.")
  .NumInput(0)
  .NumOutput(2)  // (Images, Labels)
  .AddArg("file_root",
      R"code(Path to a directory that contains the data files.

``FileReader`` supports a flat directory structure. The ``file_root`` directory contains
directories with data files. To obtain the labels, ``FileReader`` sorts directories in ``file_root``
in alphabetical order and takes an index in this order as a class label.)code",
      DALI_STRING)
  .AddOptionalArg("file_list",
      R"code(Path to a text file that contains the rows of ``filename label`` pairs,
where the filenames are relative to ``file_root``.

If left empty, ``file_root`` is traversed for subdirectories, which are only at one level
down from ``file_root``, and contain files that are associated with the same label.
When traversing subdirectories, the labels are assigned consecutive numbers.)code",
      std::string())
.AddOptionalArg("shuffle_after_epoch",
      R"code(If set to True, the reader shuffles the entire dataset after each epoch.

``stick_to_shard`` and ``random_shuffle`` are mutually exclusive.)code",
      false)
  .AddParent("LoaderBase");

}  // namespace dali
