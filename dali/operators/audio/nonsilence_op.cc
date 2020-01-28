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

#include <dali/kernels/signal/dct/dct_cpu.h>
#include <dali/pipeline/data/views.h>
#include "dali/operators/audio/nonsilence_op.h"
#include "dali/core/static_switch.h"
#include "dali/core/convert.h"

namespace dali {

DALI_SCHEMA(NonsilenceRegion)
                .DocStr(R"code(The operator performs leading and trailing silence detection in an audio buffer.<br>
This operators' behaviour can be described as:
```
def nonsilence(buffer, cutoff_value):
    begin = 0
    end = 0
    for i in range(len(buffer)):
        if buffer[i] > cutoff_value:
            begin = i
            break
    for i in range(len(buffer) - 1, -1, -1):
        if buffer[i] > cutoff_value:
            end = i
            break
    length = end - begin + 1
    return begin, length
```
`Input`: 1-D audio buffer
`Output[0]`: Begin index of nonsilent region
`Output[1] >= 0`: Length of nonsilent region<br>
If `Output[1] == 0`, `Output[0]` value is undefined
)code")
                .NumInput(1)
                .NumOutput(detail::kNumOutputs)
                .AddArg(detail::kCutoff,
                        R"code(Everything below this value will be regarded as silence)code",
                        DALI_FLOAT);

DALI_REGISTER_OPERATOR(NonsilenceRegion, NonsilenceOperatorCpu, CPU);


bool NonsilenceOperatorCpu::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const workspace_t<CPUBackend> &ws) {
return this->impl_->SetupImpl(output_desc,ws);
}

#define NONSILENCE_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float, double)  // NOLINT


void NonsilenceOperatorCpu::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);

//  TYPE_SWITCH(input.type().id(), type2id, InputType, NONSILENCE_TYPES, (
//          this->impl_->RunImplTyped<InputType>(ws);
//  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())))  // NOLINT
this->impl_->RunImplTyped<float>(ws);

}


#undef NONSILENCE_TYPES

}  // namespace dali
