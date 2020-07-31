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

DALI_REGISTER_OPERATOR(SequenceReader, SequenceReader, CPU);

DALI_SCHEMA(SequenceReader)
    .DocStr(
        R"code(Reads [Frame] sequences from a directory that represents a collection of streams.

This class expects file_root to contain a set of directories, where each directory represents
an extracted video stream. This stream is represented by one file for each frame and
lexicographically sorting the paths to frames  will give the original order of frames.
Sequences do not cross the stream boundary and only complete sequences are considered, so
there is no padding.

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
      ....)code")
    .NumInput(0)
    .NumOutput(1)  // ([Frames])
    .AddArg("file_root",
            R"code(Path to a directory containing streams (directories representing streams).)code",
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

}  // namespace dali
