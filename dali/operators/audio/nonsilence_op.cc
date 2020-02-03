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

DALI_SCHEMA(NonsilentRegion)
                .DocStr(R"code(The operator performs leading and trailing silence detection in an audio buffer.
The operator returns the beginning and length of the non-silent region by comparing short term power of the signal
with a silence cut-off threshold. The signal is consider silence when ``short_term_power_db < cutoff_db`` with::

  short_term_power_db = 10 * log10( short_term_power / reference_power )

and ``reference_power`` being typically the maximum of the signal, unless specified otherwise.

Inputs/Outputs
  **Input 0** - 1D audio buffer
  **Output 0** - Begin index of nonsilent region
  **Output 1** - Length of nonsilent region

Remarks
  - If ``Outputs[1] == 0``, ``Outputs[0]`` value is undefined
)code")
                .NumInput(1)
                .NumOutput(detail::kNumOutputs)
                .AddOptionalArg("cutoff_db",
                                R"code(The threshold [dB], below which everything is considered as silence)code",
                                -60.f)
                .AddOptionalArg("window_length", R"code(Size of a sliding window.
The sliding window is used to calculate short-term power of the signal.)code", 2048)
                .AddOptionalArg("reference_power",
                                R"code(The reference power used for converting signal to db.
If ``reference_power`` is not provided, the maximum of the signal will be used as the reference power)code",
                                0.f)
                .AddOptionalArg("reset_interval",
                                R"code(The number of samples after which the moving mean average is
recalculated to avoid loss of precision. If ``reset_interval == -1`` or the input type allows exact calculation,
the average won't be reset. The default value should fit most of the use cases.)code",
                                8192);

DALI_REGISTER_OPERATOR(NonsilentRegion, NonsilenceOperatorCpu, CPU);


bool NonsilenceOperatorCpu::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const workspace_t<CPUBackend> &ws) {
  AcquireArgs(spec_, ws);
  TypeInfo output_type;
  output_type.SetType<int>(TypeTable::GetTypeID<int>());
  TensorShape<> scalar_shape = {1};

  output_desc.resize(detail::kNumOutputs);
  for (int i = 0; i < detail::kNumOutputs; i++) {
    output_desc[i].shape = uniform_list_shape(batch_size_, scalar_shape);
    output_desc[i].type = output_type;
  }
  return true;
}


#define NONSILENCE_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float)  // NOLINT


void NonsilenceOperatorCpu::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  TYPE_SWITCH(input.type().id(), type2id, InputType, NONSILENCE_TYPES, (
          RunImplTyped<InputType>(ws);
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())))  // NOLINT
}


#undef NONSILENCE_TYPES


}  // namespace dali
