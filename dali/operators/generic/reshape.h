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

#ifndef DALI_OPERATORS_GENERIC_RESHAPE_H_
#define DALI_OPERATORS_GENERIC_RESHAPE_H_

#include <string>
#include <vector>

#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/core/tensor_view.h"

namespace dali {

template <typename Backend>
class Reshape : public Operator<Backend> {
 public:
  using Base = Operator<Backend>;
  using Workspace = workspace_t<Backend>;

  explicit Reshape(const OpSpec &spec_);

  bool CanInferOutputs() const override {
    // Return false, because we specifically don't want the executor to allocate
    // the storage for the output - even though we can infer the shape.
    return false;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;

  void RunImpl(Workspace &ws) override;
 
 protected:
  virtual void CalculateOutputShape(const Workspace &ws);
  void CheckSrcDims(const Workspace &ws);
  
  std::vector<int> src_dims_;

 protected:
  virtual void CalculateOutputShape(const Workspace &ws);
  void CheckSrcDims(const Workspace &ws);

  std::vector<int> src_dims_;

 private:
  inline const std::string &OpName() const {
    return this->spec_.name();
  }

  TensorListShape<> input_shape_, output_shape_;
  TensorShape<> uniform_shape_;
  std::vector<float> rel_uniform_shape_;
  TensorLayout layout_;
  bool use_layout_ = false;
  bool use_rel_shape_ = false;
  int wildcard_dim_ = -1;
  DALIDataType output_type_id_ = DALI_NO_TYPE;
  const TypeInfo *output_type_ = nullptr;

  enum class ShapeSource {
    None,
    Input,
    Arg,
    ArgInput
  };
  ShapeSource shape_source_ = ShapeSource::None;
<<<<<<< HEAD
<<<<<<< HEAD
=======
  bool use_rel_shape_ = false;
  int wildcard_dim_ = -1;
  DALIDataType output_type_id_ = DALI_NO_TYPE;
  const TypeInfo *output_type_ = nullptr;
>>>>>>> Adding src_dims to reshape operator
=======
>>>>>>> Code review suggestions

  template <typename TensorListLike>
  void ShapeFromInput(const TensorListLike &tl, bool relative);

  template <typename Extent>
  void ShapeFromInput(const TensorListView<StorageCPU, Extent> &shape);

  TensorLayout GetOutputLayout(const Workspace &ws) const;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_RESHAPE_H_
