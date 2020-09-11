// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/reader/nemo_asr_reader_op.h"

namespace dali {

DALI_REGISTER_OPERATOR(NemoAsrReader, NemoAsrReader, CPU);

DALI_SCHEMA(NemoAsrReader)
  .NumInput(0)
  .NumOutput(1)
  .DocStr(R"code(Read automatic speech recognition (ASR) data (audio, text) from a 
NVIDIA NeMo compatible manifest.

Example manifest file::

    {"audio_filepath": "path/to/audio1.wav", "duration": 3.45, "text": "this is a nemo tutorial"}
    {"audio_filepath": "path/to/audio1.wav", "offset": 3.45, "duration": 1.45, "text": "same audio file but using offset"}
    {"audio_filepath": "path/to/audio2.wav", "duration": 3.45, "text": "third transcript in this example"}

This reader produces between 1 and 3 outputs:

- Decoded audio data: float, shape=``(audio_length,)``

- (optional, if ``read_sample_rate=True``) Audio sample rate: float, shape=``(1,)``

- (optional, if ``read_text=True``) Transcript text as a null terminated string: uint8, shape=``(text_len + 1,)``

)code")
  .AddArg("manifest_filepath",
    "Path to the manifest file",
    DALI_STRING)
  .AddOptionalArg("read_sample_rate",
    "Whether to output the sample rate for each sample as a separate output",
    true)
  .AddOptionalArg("read_text",
    "Whether to output the transcript text for each sample as a separate output",
    true)
  .AddOptionalArg("shuffle_after_epoch",
    "If true, reader shuffles whole dataset after each epoch",
    false)
  .AddOptionalArg("sample_rate",
    "If specified, the target sample rate, in Hz, to which the audio is resampled.",
    -1.0f)
  .AddOptionalArg("quality",
    "Resampling quality, 0 is lowest, 100 is highest.\n"
    "0 corresponds to 3 lobes of the sinc filter; "
    "50 gives 16 lobes and 100 gives 64 lobes.",
     50.0f)
  .AddOptionalArg("downmix",
    "If True, downmix all input channels to mono. "
    "If downmixing is turned on, decoder will produce always 1-D output",
    true)
  .AddOptionalArg("dtype",
    "Type of the output data. Supports types: `INT16`, `INT32`, `FLOAT`",
    DALI_FLOAT)
  .AddOptionalArg("max_duration",
    "It a value greater than 0 is provided, it specifies the maximum allowed duration, "
    "in seconds, of the audio samples.\n"
    "Samples with a duration longer than this value will be ignored.\n",
    0.0f)
  .AdditionalOutputsFn([](const OpSpec& spec) {
    return static_cast<int>(spec.GetArgument<bool>("read_sample_rate"))
         + static_cast<int>(spec.GetArgument<bool>("read_text"));
  })
  .AddParent("LoaderBase");

}  // namespace dali
