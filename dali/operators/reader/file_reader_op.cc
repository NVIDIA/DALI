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

DALI_REGISTER_OPERATOR(readers__File, FileReader, CPU);

DALI_SCHEMA(readers__File)
  .DocStr(R"(Reads file contents and returns file-label pairs.

This operator can be used in the following modes:

1. Listing files from a directory, assigning labels based on subdirectory structure.

In this mode, the directory indicated in ``file_root`` argument should contain one or more
subdirectories. The files in these subdirectories are listed and assigned labels based on
lexicographical order of the subdirectory.

For example, this directory structure::

  <file_root>/0/image0.jpg
  <file_root>/0/world_map.jpg
  <file_root>/0/antarctic.png
  <file_root>/1/cat.jpeg
  <file_root>/1/dog.tif
  <file_root>/2/car.jpeg
  <file_root>/2/truck.jp2

will yield the following outputs::

  <contents of 0/image0.jpg>        0
  <contents of 0/world_map.jpg>     0
  <contents of 0/antarctic.png>     0
  <contents of 1/cat.jpeg>          1
  <contents of 1/dog.tif>           1
  <contents of 2/car.jpeg>          2
  <contents of 2/truck.jp2>         2

2. Use file names and labels stored in a text file.

``file_list`` argument points to a file which contains one file name and label per line.
Example::

  dog.jpg 0
  cute kitten.jpg 1
  doge.png 0

The file names can contain spaces in the middle, but cannot contain trailing whitespace.

3. Use file names and labels provided as a list of strings and integers, respectively.

As with other readers, the (file, label) pairs returned by this operator can be randomly shuffled
and various sharding strategies can be applied. See documentation of this operator's arguments
for details.
)")
  .NumInput(0)
  .NumOutput(2)  // (Images, Labels)
  .AddOptionalArg<string>("file_root",
      R"(Path to a directory that contains the data files.

If not using ``file_list`` or ``files``, this directory is traversed to discover the files.
``file_root`` is required in this mode of operation.)",
      nullptr)
  .AddOptionalArg<string>("file_list",
      R"(Path to a text file that contains one whitespace-separated ``filename label``
pair per line. The filenames are relative to the location of that file or to ``file_root``,
if specified.

This argument is mutually exclusive with ``files``.)", nullptr)
.AddOptionalArg("shuffle_after_epoch",
      R"(If set to True, the reader shuffles the entire dataset after each epoch.

``stick_to_shard`` and ``random_shuffle`` cannot be used when this argument is set to True.)",
      false)
  .AddOptionalArg<vector<string>>("files", R"(A list of file paths to read the data from.

If ``file_root`` is provided, the paths are treated as being relative to it.
When using ``files``, the labels are taken from ``labels`` argument or, if it was not supplied,
contain indices at which given file appeared in the ``files`` list.

This argument is mutually exclusive with ``file_list``.)", nullptr)
  .AddOptionalArg<vector<int>>("labels", R"(Labels accompanying contents of files listed in
``files`` argument.

If not used, sequential 0-based indices are used as labels)", nullptr)
  .AddParent("LoaderBase");


// Deprecated alias
DALI_REGISTER_OPERATOR(FileReader, FileReader, CPU);

DALI_SCHEMA(FileReader)
    .DocStr("Legacy alias for :meth:`readers.file`.")
    .NumInput(0)
    .NumOutput(2)  // (Images, Labels)
    .AddParent("readers__File")
    .MakeDocPartiallyHidden()
    .Deprecate(
        "readers__File",
        R"code(In DALI 1.0 all readers were moved into a dedicated :mod:`~nvidia.dali.fn.readers`
submodule and renamed to follow a common pattern. This is a placeholder operator with identical
functionality to allow for backward compatibility.)code");  // Deprecated in 1.0;

}  // namespace dali
