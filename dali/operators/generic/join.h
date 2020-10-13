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

#include <string>
#include <vector>
#include "dali/core/any.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/common/join/tensor_join_cpu.h"
#include "dali/kernels/common/join/tensor_join_gpu.h"

namespace dali {

template <typename Backend, bool new_axis>
class TensorJoin : public Operator<Backend> {
 public:
  explicit TensorJoin(const OpSpec &spec) : Operator<Backend>(spec) {
    has_axis_ = spec.HasArgument("axis");
    has_axis_name_ = spec.HasArgument("axis_name");
    if (!new_axis) {
      DALI_ENFORCE(!(has_axis_ && has_axis_name_),
        "Arguments ``axis`` and ``axis_name`` cannot be used together.");
    }

    if (has_axis_)
      axis_arg_ = spec.GetArgument<int>("axis");
    if (has_axis_name_) {
      auto axis_name_str = spec.GetArgument<std::string>("axis_name");
      DALI_ENFORCE(axis_name_str.length() == 1, make_string("``axis_name``"
        " must be a single character; got ", axis_name_str));
      axis_name_arg_ = axis_name_str[0];
    }
  }

  using Storage = detail::storage_tag_map_t<Backend>;

  using Operator<Backend>::Operator;

  bool CanInferOutputs() const override { return true; }
  void RunImpl(workspace_t<Backend> &ws) override;
  bool SetupImpl(vector<OutputDesc> &outputs, const workspace_t<Backend> &ws) override;

 protected:
  template <typename T>
  auto &inputs() {
    using RetType = vector<TensorListView<Storage, const T>>;
    if (RetType *inp = any_cast<RetType>(&inputs_))
      return *inp;
    inputs_ = RetType();
    return any_cast<RetType&>(inputs_);
  }

  template <typename T>
  void SetupTyped(TensorListShape<> &output_shape, const workspace_t<Backend> &ws);

  template <typename T>
  void RunTyped(const TensorListView<Storage, T> &out, HostWorkspace &ws);

  template <typename T>
  void RunTyped(const TensorListView<Storage, T> &out, DeviceWorkspace &ws);

  void GetInputLayout(const workspace_t<Backend> &ws);
  void SetupAxis();
  void SetOutputLayout(const workspace_t<Backend> &ws);

  any inputs_;
  kernels::KernelManager kmgr_;
  int axis_ = -1;
  TensorLayout input_layout_, output_layout_;

  bool has_axis_ = false, has_axis_name_ = false;
  int axis_arg_ = 0;
  int copy_idx_ = -1;
  char axis_name_arg_ = 0;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_JOIN_H_
