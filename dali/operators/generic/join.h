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

#ifndef DALI_OPERATORS_GENERIC_JOIN_H_
#define DALI_OPERATORS_GENERIC_JOIN_H_

#include "dali/kernels/common/join/tensor_join_cpu.h"
#include "dali/kernels/common/join/tensor_join_gpu.h"

namespace dali {

template <typename Backend, bool new_axis>
class TensorJoin<Backend> : public Operator<Backend> {
 public:
  using Operator<Backend>::Operator;
  bool SetupImpl()
  bool CanInferOutputs() const override { return true; }
 protected:
  void CollectInputs(const workspace_t<Backend> &ws);

  SmallVector<TensorListView<void>> inputs_;
  TypeInfo output_type_;
  KernelManager mgr_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_JOIN_H_
