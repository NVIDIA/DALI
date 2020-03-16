// Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_RANDOM_ONE_HOT_H
#define DALI_OPERATORS_RANDOM_ONE_HOT_H

#include <vector>

#include "dali/pipeline/operator/operator.h"
#include "dali/kernels/kernel_params.h"
#include "dali/core/tensor_view.h"
#include "dali/core/static_switch.h"

#define ONE_HOT_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float, double)  // NOLINT

namespace dali {

namespace detail {
template <typename Out, typename In, int ndims = 1>
void DoOneHot(kernels::OutTensorCPU<Out, ndims> &out, 
              const kernels::InTensorCPU<In, ndims> &in,
              int depth, int on_value, int off_value) {
  auto input = in.data;
  auto output = out.data;
  for (int i = 0; i < depth; ++i) {
    if (i == static_cast<int>(*input)) {
      output[i] = on_value;
    } else {
      output[i] = off_value;
    }
  }
}
} // namespace detail

class OneHot : public Operator<CPUBackend> {
 public:
  inline explicit OneHot(const OpSpec &spec)
      : Operator<CPUBackend>(spec), depth_(spec.GetArgument<int64_t>("depth")),
        output_type_(spec.GetArgument<std::remove_const_t<decltype(this->output_type_)>>(
                     arg_names::kDtype)),
        on_value_(spec.GetArgument<int64_t>("on_value")),
        off_value_(spec.GetArgument<int64_t>("off_value")) {}

  inline ~OneHot() override = default;

  DISABLE_COPY_MOVE_ASSIGN(OneHot);

  USE_OPERATOR_MEMBERS();
  using Operator<CPUBackend>::RunImpl;

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const HostWorkspace &ws) override;
  void RunImpl(HostWorkspace &ws) override;


  int depth_;
  const DALIDataType output_type_;
  int on_value_;
  int off_value_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_ONE_HOT_H