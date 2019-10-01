// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_UTIL_RESHAPE_H_
#define DALI_PIPELINE_OPERATORS_UTIL_RESHAPE_H_

#include <vector>

#include "dali/pipeline/operators/operator.h"
#include "dali/kernels/tensor_view.h"

namespace dali {

template <typename Backend>
class Reshape : public Operator<Backend> {
 public:
  using Base = Operator<Backend>;
  using Workspace = workspace_t<Backend>;

  explicit Reshape(const OpSpec &spec_);

  bool CanInferOutputs() const override { return false; }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;

  void RunImpl(Workspace &ws) override;

 private:
  kernels::TensorListShape<> input_shape_, output_shape_;
  kernels::TensorShape<> uniform_shape_;
  TensorLayout layout_;
  enum class ShapeSource {
    None,
    Input,
    Arg,
    ArgInput
  } shape_source_ = ShapeSource::None;
  void CalculateOutputShape(const Workspace &ws);

  template <typename TensorListLike>
  void ShapeFromInput(const TensorListLike &tl);

  template <typename Integer>
  void ShapeFromInput(const kernels::TensorListView<kernels::StorageCPU, Integer> &shape);

  TensorLayout GetOutputLayout(const Workspace &ws) const;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_UTIL_RESHAPE_H_
