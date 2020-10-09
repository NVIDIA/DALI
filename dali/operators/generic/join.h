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

#include "dali/core/any.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/common/join/tensor_join_cpu.h"
#include "dali/kernels/common/join/tensor_join_gpu.h"

namespace dali {

template <typename Backend, bool new_axis>
class TensorJoin : public Operator<Backend> {
 public:
  using Storage = detail::storage_tag_map_t<Backend>;

  using Operator<Backend>::Operator;

  bool CanInferOutputs() const override { return true; }
  void RunImpl(workspace_t<Backend> &ws);
  bool SetupImpl(vector<OutputDesc> &outputs, const workspace_t<Backend> &ws) override;
 protected:

  template <typename T>
  auto &inputs() {
    using RetType = vector<const TensorListView<Storage, const T>>;
    if (!inputs_.is_type<RetType>())
        inputs_ = RetType();
    return any_cast<RetType&>(inputs_);
  }

  template <typename T>
  void SetupTyped(TensorListShape<> &output_shape, const workspace_t<Backend> &ws);

  template <typename T>
  void RunTyped(const TensorListView<Storage, T> &out, HostWorkspace &ws);

  template <typename T>
  void RunTyped(const TensorListView<Storage, T> &out, DeviceWorkspace &ws);

  int axis_ = -1;
  TensorLayout GetLayout(const workspace_t<Backend> &ws);

  any inputs_;
  kernels::KernelManager kmgr_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_JOIN_H_
