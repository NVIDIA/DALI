// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_GENERIC_CONSTANT_H_
#define DALI_OPERATORS_GENERIC_CONSTANT_H_

#include <vector>

#include "dali/pipeline/operator/operator.h"
#include "dali/core/tensor_view.h"
#include "dali/core/static_switch.h"

#define CONSTANT_OP_SUPPORTED_TYPES \
  (bool, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, float16)

namespace dali {

template <typename Backend>
class Constant : public Operator<Backend> {
 public:
  using Base = Operator<Backend>;
  using Workspace = workspace_t<Backend>;

  explicit Constant(const OpSpec &spec) : Operator<Backend>(spec) {
    bool has_shape = spec.ArgumentDefined("shape");
    spec.TryGetRepeatedArgument<int>(shape_arg_, "shape");
    output_type_ = spec.GetArgument<DALIDataType>("dtype");
    if (spec.HasArgument("fdata")) {
      DALI_ENFORCE(!spec.HasArgument("idata"), "Constant node: `fdata` and `idata` arguments are "
        "mutually exclusive");
      fdata_ = spec.GetRepeatedArgument<float>("fdata");
      if (!has_shape) {
        shape_arg_ = { static_cast<int>(fdata_.size()) };
      } else {
        DALI_ENFORCE(fdata_.size() == static_cast<size_t>(volume(shape_arg_)) || fdata_.size() == 1,
          "The number of values does not match the shape specified");
      }

      if (!spec.HasArgument("dtype"))
        output_type_ = DALI_FLOAT;
    } else {
      DALI_ENFORCE(spec.HasArgument("idata"),
          "Constant node: either `fdata` or `idata` must be present.");

      if (!spec.HasArgument("dtype"))
        output_type_ = DALI_INT32;

      idata_ = spec.GetRepeatedArgument<int>("idata");
      if (!has_shape) {
        shape_arg_ = { static_cast<int>(idata_.size()) };
      } else {
        DALI_ENFORCE(idata_.size() == static_cast<size_t>(volume(shape_arg_)) || idata_.size() == 1,
          "The number of values does not match the shape specified");
      }
    }
    layout_ = spec.GetArgument<TensorLayout>("layout");
    if (!layout_.empty()) {
      DALI_ENFORCE(layout_.size() == static_cast<int>(shape_arg_.size()), make_string(
        "Constant node: The requested layout \"", layout_, "\" has dimensionality which is "
        "incompatible with the requested output shape."));
    }
  }

  bool CanInferOutputs() const override {
    // Return false, because we specifically don't want the executor to allocate
    // the storage for the output - even though we can infer the shape.
    return false;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    output_desc.resize(1);
    if (max_output_shape_.empty()) {
      max_output_shape_ = uniform_list_shape(max_batch_size_, shape_arg_);
      output_.Reset();
    }
    output_shape_ = max_output_shape_;
    output_shape_.resize(ws.GetRequestedBatchSize(0));
    output_desc[0] = {output_shape_, TypeTable::GetTypeInfo(output_type_)};
    return false;
  }

  void RunImpl(Workspace &ws) override;

 private:
  USE_OPERATOR_MEMBERS();
  std::vector<int> shape_arg_;
  std::vector<int> idata_;
  std::vector<float> fdata_;
  TensorListShape<> output_shape_;
  TensorListShape<> max_output_shape_;
  TensorLayout layout_;
  DALIDataType output_type_;
  using storage_t = std::conditional_t<std::is_same<Backend, CPUBackend>::value,
    TensorVector<CPUBackend>, TensorList<GPUBackend>>;
  storage_t output_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_CONSTANT_H_
