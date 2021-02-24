// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/reader/sequence_reader_op.h"

namespace dali {

void SequenceReader::RunImpl(SampleWorkspace &ws) {
  parser_->Parse(GetSample(ws.data_idx()), &ws);
}

DALI_REGISTER_OPERATOR(readers__Sequence, SequenceReader, CPU);

DALI_SCHEMA(readers__Sequence)
    .DocStr(
        R"code(Reads [Frame] sequences from a directory representing a collection of streams.

This operator expects ``file_root`` to contain a set of directories, where each directory represents
an extracted video stream. This stream is represented by one file for each frame,
sorted lexicographically. Sequences do not cross the stream boundary and only complete sequences
are considered, so there is no padding.

Example directory structure::

  - file_root
    - 0
      - 00001.png
      - 00002.png
      - 00003.png
      - 00004.png
      - 00005.png
      - 00006.png
      ....

    - 1
      - 00001.png
      - 00002.png
      - 00003.png
      - 00004.png
      - 00005.png
      - 00006.png
      ....

.. note::
  This operator is an analogue of VideoReader working on video frames extracted as separate images.
  It's main purpose is for test baseline. For regular usage, the VideoReader is
  the recommended approach.)code")
    .NumInput(0)
    .NumOutput(1)  // ([Frames])
    .AddArg("file_root",
            R"code(Path to a directory containing streams, where the directories
represent streams.)code",
            DALI_STRING)
    .AddArg("sequence_length",
            R"code(Length of sequence to load for each sample.)code", DALI_INT32)
    .AddOptionalArg("step",
                    R"code(Distance between first frames of consecutive sequences.)code", 1, false)
    .AddOptionalArg("stride",
                    R"code(Distance between consecutive frames in a sequence.)code", 1, false)
    .AddOptionalArg("image_type",
                    R"code(The color space of input and output image.)code", DALI_RGB, false)
    .AddParent("LoaderBase")
    .AllowSequences();


// Deprecated alias
DALI_REGISTER_OPERATOR(SequenceReader, SequenceReader, CPU);

DALI_SCHEMA(SequenceReader)
    .DocStr("Legacy alias for :meth:`readers.sequence`.")
    .NumInput(0)
    .NumOutput(1)  // ([Frames])
    .AllowSequences()
    .AddParent("readers__Sequence")
    .MakeDocPartiallyHidden()
    .Deprecate(
        "readers__Sequence",
        R"code(In DALI 1.0 all readers were moved into a dedicated :mod:`~nvidia.dali.fn.readers`
submodule and renamed to follow a common pattern. This is a placeholder operator with identical
functionality to allow for backward compatibility.)code");  // Deprecated in 1.0;

}  // namespace dali
