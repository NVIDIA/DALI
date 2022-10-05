// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/decoder/inflate/inflate.h"

namespace dali {

DALI_SCHEMA(experimental__Inflate)
    .DocStr(R"code(Inflate the binary input using specified decompression algorithm.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddArg("shape", "The shape of the output (decoded) sample or frame.", DALI_INT_VEC, true)
    .AddOptionalTypeArg("dtype", R"code(Output data type.)code", DALI_UINT8)
    .AddOptionalArg("frame_offset",
                    "Required if the input sample is a sequence of encoded samples.",
                    std::vector<int>{}, true)
    .AddOptionalArg("frame_size",
                    "Ignored if input is not a sequence. If the input is a sequence, signifies the "
                    "sizes of corresponding encoded frames, if not provided consecutive frames are "
                    "assumed to be densely packed and sizes are inferred from ``frame_offset``.",
                    std::vector<int>{}, true)
    .AddOptionalArg(
        "algorithm",
        R"code(Algorithm to be used to decode the data. Currently only ``LZ4`` is supported.)code",
        "LZ4");

}  // namespace dali
