// Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_BUILTIN_COPY_H_
#define DALI_PIPELINE_OPERATOR_BUILTIN_COPY_H_

#include <cstring>
#include <vector>
#include <type_traits>

#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/kernels/common/scatter_gather.h"

namespace dali {

template <typename Backend>
class Copy : public StatelessOperator<Backend> {
 public:
  inline explicit Copy(const OpSpec &spec) :
    StatelessOperator<Backend>(spec), scatter_gather_(kMaxSizePerBlock) {}

  inline ~Copy() override = default;

  DISABLE_COPY_MOVE_ASSIGN(Copy);

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    output_desc.resize(1);
    const auto &input = ws.Input<Backend>(0);
    output_desc[0].type = input.type();
    output_desc[0].shape = input.shape();
    return true;
  }

  void RunImpl(Workspace &ws) override {
    auto &input = ws.Input<Backend>(0);
    auto data_type_size = input.type_info().size();
    auto &output = ws.Output<Backend>(0);
    output.SetLayout(input.GetLayout());
    for (int i = 0; i < input.num_samples(); i++) {
      auto tensor_shape = input.tensor_shape(i);
      auto tensor_size = volume(tensor_shape);
      scatter_gather_.AddCopy(output.raw_mutable_tensor(i), input.raw_tensor(i),
                              tensor_size * data_type_size);
    }
    RunCopies(ws);
  }

  void RunCopies(Workspace &ws);

  std::conditional_t<
      std::is_same<Backend, CPUBackend>::value,
      kernels::ScatterGatherCPU,
      kernels::ScatterGatherGPU> scatter_gather_;
  // 256 kB per block for GPU
  static constexpr size_t kMaxSizePerBlock =
      std::is_same<Backend, CPUBackend>::value ? kernels::ScatterGatherCPU::kAnyBlockSize : 1 << 18;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_COPY_H_
