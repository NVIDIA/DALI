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
  .DocStr(R"code(Performs leading and trailing silence detection in an audio buffer.

The operator returns the beginning and length of the non-silent region by comparing the
short term power calculated for ``window_length`` of the signal with a silence cut-off threshold.
The signal is considered to be silent when the ``short_term_power_db`` is less than
the ``cutoff_db``. where::

  short_term_power_db = 10 * log10( short_term_power / reference_power )

Unless specified otherwise, ``reference_power`` is the maximum power of the signal.

Inputs and outputs:

* **Input 0** - 1D audio buffer.
* **Output 0** - Index of the first sample in the nonsilent region.
* **Output 1** - Length of nonsilent region.

.. note::
  If ``Outputs[1] == 0``,  the value in ``Outputs[0]`` is undefined.)code")
  .NumInput(1)
  .NumOutput(detail::kNumOutputs)
  .AddOptionalArg("cutoff_db",
                  R"code(The threshold, in dB, below which the signal is considered silent.)code",
                  -60.f)
  .AddOptionalArg("window_length", R"code(Size of the sliding window used to calculate of
the short-term power of the signal.)code", 2048)
  .AddOptionalArg("reference_power",
                  R"code(The reference power that is used to convert the signal to dB.

If a value is not provided, the maximum power of the signal will be used as the reference.)code",
                  0.f)
  .AddOptionalArg("reset_interval",
                  R"code(The number of samples after which the moving mean average is recalculated
to avoid loss of precision.

If ``reset_interval == -1``, or the input type allows exact calculation, the average will not be
reset. The default value can be used for most of the use cases.)code",
                  8192);

DALI_REGISTER_OPERATOR(NonsilentRegion, NonsilenceOperatorCpu, CPU);


bool NonsilenceOperatorCpu::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const workspace_t<CPUBackend> &ws) {
  AcquireArgs(spec_, ws);
  TypeInfo output_type;
  output_type.SetType<int>(TypeTable::GetTypeID<int>());
  TensorShape<> scalar_shape = {};
  auto curr_batch_size = ws.GetInputBatchSize(0);

  output_desc.resize(detail::kNumOutputs);
  for (int i = 0; i < detail::kNumOutputs; i++) {
    output_desc[i].shape = uniform_list_shape(curr_batch_size, scalar_shape);
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
