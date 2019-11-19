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

#include <dali/pipeline/operator/op_schema.h>
#include "audio_decoder_op.h"

namespace dali {

//TODO in docs:
// 1. Supported output: int16, int32, float

DALI_SCHEMA(AudioDecoder)
                .DocStr(R"code(Decode audio data.
Output: (Metadata, AudioData))code")
                .NumInput(1)
                .NumOutput(detail::kNumOutputs)
                .AddOptionalArg(detail::kOutputTypeName, "output_type", DALI_INT16);

DALI_REGISTER_OPERATOR(AudioDecoder, AudioDecoderCpu, CPU);

}