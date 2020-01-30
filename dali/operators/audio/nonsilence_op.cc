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

#include "dali/core/static_switch.h"
#include "dali/operators/audio/nonsilence_op.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(NonsilenceRegion)
                .DocStr(R"code(The operator performs leading and trailing silence detection in an audio buffer.<br>
This operators' behaviour can be described as::

def nonsilence(buffer, cutoff_value, top_db, ref_db):
    buffer = ToDecibels(buffer, top_db, ref_db)
    begin = end = 0
    for i in [0 .. len(buffer)):
        if buffer[i] > cutoff_value:
            begin = i; break
    for i in [len(buffer)-1 .. begin]:
        if buffer[i] > cutoff_value:
            end = i; break
    length = end - begin + 1
    return begin, length

`Input`: 1-D audio buffer
`Output[0]`: Begin index of nonsilent region
`Output[1] >= 0`: Length of nonsilent region
If `Output[1] == 0`, `Output[0]` value is undefined
)code")
                .NumInput(1)
                .NumOutput(detail::kNumOutputs)
                .AddOptionalArg("cutoff_db",
                                R"code(The threshold [dB], below which everything is considered as silence)code",
                                60.f)
                .AddOptionalArg("window_length", R"code(Size of analysing window)code", 2048)
                .AddOptionalArg("reference_db",
                                R"code(The reference power. If `reference_max` is `True`, this value is ignored)code",
                                1.f)
                .AddOptionalArg("reference_max",
                                R"code(Is `True`, the maximum of the signal will be used as the reference power (instead of `reference_db`))code",
                                false)
                .AddOptionalArg("reset_interval",
                                R"code(The number of samples after which the moving mean average is recalculated to avoid loss of precision. Ignored if the input type allows exact calculation)code",
                                -1);

DALI_REGISTER_OPERATOR(NonsilenceRegion, NonsilenceOperatorCpu, CPU);


bool NonsilenceOperatorCpu::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const workspace_t<CPUBackend> &ws) {
  AcquireArgs(spec_, ws);
  return this->impl_->SetupImpl(output_desc, ws);
}


#define NONSILENCE_TYPES (uint8_t,/* int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t,*/ float/*, double*/)  // NOLINT

void NonsilenceOperatorCpu::RunImpl(workspace_t<CPUBackend> &ws) {
//  const auto &input = ws.template InputRef<CPUBackend>(0);
//  TYPE_SWITCH(input.type().id(), type2id, InputType, NONSILENCE_TYPES, (
//          this->impl_->RunImplTyped<InputType>(ws);
//  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())))  // NOLINT

  this->impl_->RunImplTyped<uint8_t >(ws);
}

#undef NONSILENCE_TYPES




}  // namespace dali
