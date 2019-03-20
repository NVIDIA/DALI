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

#include "dali/pipeline/operators/reader/sequence_reader_op.h"

namespace dali {

void SequenceReader::RunImpl(SampleWorkspace* ws, const int i) {
  parser_->Parse(GetSample(ws->data_idx()), ws);
}

DALI_REGISTER_OPERATOR(SequenceReader, SequenceReader, CPU);

DALI_SCHEMA(SequenceReader)
    .DocStr(
        R"code(Read [Frame] sequences from a directory representing collection of streams.
Expects file_root to contain set of directories, each of them represents one extracted video
stream. Extracted video stream is represented by one file for each frame, sorting the paths to
frames lexicographically should give the original order of frames.
Sequences do not cross stream boundary and only full sequences are considered - there is no padding.
Example:
> file_root
  > 0
    > 00001.png
    > 00002.png
    > 00003.png
    > 00004.png
    > 00005.png
    > 00006.png
    ....
  > 1
    > 00001.png
    > 00002.png
    > 00003.png
    > 00004.png
    > 00005.png
    > 00006.png
    ....)code")
    .NumInput(0)
    .NumOutput(1)  // ([Frames])
    .AddArg("file_root",
            R"code(Path to a directory containing streams (directories representing streams).)code",
            DALI_STRING)
    .AddArg("sequence_length",
            R"code(Length of sequence to load for each sample)code", DALI_INT32)
    .AddOptionalArg("step",
                    R"code(Distance between first frames of consecutive sequences)code", 1, false)
    .AddOptionalArg("stride",
                    R"code(Distance between consecutive frames in sequence)code", 1, false)
    .AddOptionalArg("image_type",
                    R"code(The color space of input and output image)code", DALI_RGB, false)
    .AddParent("LoaderBase");

}  // namespace dali
