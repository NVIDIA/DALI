// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_GENERIC_EXPAND_DIMS_H_
#define DALI_OPERATORS_GENERIC_EXPAND_DIMS_H_

#include <string>
#include <vector>

#include "dali/operators/generic/reshape.h"
#include "dali/core/tensor_view.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class ExpandDims : public Reshape<Backend> {
 public:
  using Base = Operator<Backend>;
  using Workspace = workspace_t<Backend>;

  explicit ExpandDims(const OpSpec &spec_);

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;
 private:
  void GenerateSrcDims(const Workspace &ws);

  SmallVector<int, 6> axes_;
  TensorLayout new_axis_names_;
  bool use_new_axis_names_arg_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_EXPAND_DIMS_H_
