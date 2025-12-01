// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include "dali/operators/generic/cast.h"
#include "dali/core/static_switch.h"

namespace dali {

class CastCPU : public Cast<CPUBackend> {
 public:
  explicit CastCPU(const OpSpec &spec) : Cast<CPUBackend>{spec} {}

  void RunImpl(Workspace &ws) override;

  ~CastCPU() override = default;

 private:
  USE_OPERATOR_MEMBERS();
};

template <typename OType, typename IType>
inline void CpuHelper(OType *out, const IType *in, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    out[i] = ConvertSat<OType>(in[i]);
  }
}

void CastCPU::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<CPUBackend>(0);
  const auto &input_shape = input.shape();
  auto &output = ws.Output<CPUBackend>(0);
  output.SetLayout(input.GetLayout());

  auto num_samples = input_shape.num_samples();

  auto &tp = ws.GetThreadPool();
  TYPE_SWITCH(output.type(), type2id, OType, CAST_ALLOWED_TYPES, (
    TYPE_SWITCH(input.type(), type2id, IType, CAST_ALLOWED_TYPES, (

      for (int sample_id = 0; sample_id < num_samples; sample_id++) {
        auto *out = output.mutable_tensor<OType>(sample_id);
        const auto *in = input.tensor<IType>(sample_id);
        auto size = input_shape.tensor_size(sample_id);
        tp.AddWork([out, in, size](int thread_id) { CpuHelper<OType, IType>(out, in, size); },
                   size);
      }

    ), DALI_FAIL(make_string("Invalid input type: ", input.type())););  // NOLINT(whitespace/parens)
  ), DALI_FAIL(make_string("Invalid output type", output.type())););  // NOLINT(whitespace/parens)
  tp.RunAll();
}

DALI_REGISTER_OPERATOR(Cast, CastCPU, CPU);
DALI_REGISTER_OPERATOR(CastLike, CastCPU, CPU);

DALI_SCHEMA(Cast)
    .DocStr("Cast a tensor to a different type.")
    .NumInput(1)
    .NumOutput(1)
    .AllowSequences()
    .SupportVolumetric()
    .AddTypeArg("dtype", R"code(Output data type.)code");

DALI_SCHEMA(CastLike)
    .DocStr("Cast the first tensor to the type of the second tensor.")
    .NumInput(2)
    .InputDevice(1, InputDevice::Metadata)
    .NumOutput(1)
    .AllowSequences()
    .SupportVolumetric();

}  // namespace dali
