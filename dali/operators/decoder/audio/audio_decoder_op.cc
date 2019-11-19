// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/decoder/audio/audio_decoder_op.h"
#include "dali/pipeline/operator/op_schema.h"

namespace dali {

DALI_SCHEMA(AudioDecoder)
                .DocStr(R"code(Decode audio data.
This operator is a generic way of handling encoded data in DALI.
It supports most of well-known audio formats (wav, flac, ogg).

This operator produces two outputs:
output[0]: batch of decoded data
output[1]: batch of sampling rates [Hz]

Data in the output correspond to each other.
On the event more metadata will appear, we reserve a right to change this behaviour.)code")
                .NumInput(1)
                .NumOutput(detail::kNumOutputs)
                .AddOptionalArg(
                        detail::kOutputTypeName,
                        "Type of the output data. Supports types: `INT16`, `INT32`, `FLOAT`",
                        DALI_INT16);

DALI_REGISTER_OPERATOR(AudioDecoder, AudioDecoderCpu, CPU);

}  // namespace dali
