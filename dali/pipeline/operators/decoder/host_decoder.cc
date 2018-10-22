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

#include "dali/pipeline/operators/decoder/host_decoder.h"

namespace dali {

DALI_REGISTER_OPERATOR(HostDecoder, HostDecoder, CPU);

DALI_SCHEMA(HostDecoder)
    .DocStr(R"code(Decode images on the host using OpenCV.
When applicable, it will pass execution to faster, format-specific decoders (like libjpeg-turbo).
Output of the decoder is in `HWC` ordering.
In case of samples being singular images expects one input, for sequences ([frames], metadata)
pair is expected, and decode_sequences set to true.)code")
    .NumInput(1, 2)
    .NumOutput(1)
    .AddOptionalArg("output_type", R"code(The color space of output image.)code", DALI_RGB)
    .AddOptionalArg("decode_sequences", R"code(Is input a sequence of frames.)code", false);

}  // namespace dali
