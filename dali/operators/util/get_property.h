// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_UTIL_GET_PROPERTY_H_
#define DALI_OPERATORS_UTIL_GET_PROPERTY_H_

#include <memory>
#include <string>
#include <vector>
#include "dali/pipeline/data/type_traits.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {
namespace detail {

/**
 * Base class for a property of the Tensor.
 * @tparam Backend Backend of the operator
 * @tparam BatchContainer TensorList<GPUBackend> or TensorVector<CPUBackend>
 */
template <typename Backend, typename BatchContainer = batch_container_t<Backend>>
struct Property {
  Property() = default;
  virtual ~Property() = default;

  /**
   * @return The shape of the tensor containing the property, based on the input to the operator
   */
  virtual TensorListShape<> GetShape(const BatchContainer & input) = 0;

  /**
   * @return The type of the tensor containing the property, based on the input to the operator
   */
  virtual DALIDataType GetType(const BatchContainer & input) = 0;

  /**
   * This function implements filling the output of the operator. Its implementation should
   * be similar to any RunImpl function of the operator.
   */
  virtual void FillOutput(workspace_t<Backend> &) = 0;
};

}  // namespace detail

template <typename Backend>
class GetProperty : public Operator<Backend> {
 public:
  explicit GetProperty(const OpSpec &spec);

  ~GetProperty() override = default;
  DISABLE_COPY_MOVE_ASSIGN(GetProperty);

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    const auto &input = ws.template Input<Backend>(0);
    output_desc.resize(1);
    output_desc[0].shape = property_->GetShape(input);
    output_desc[0].type = property_->GetType(input);
    return true;
  }

  void RunImpl(workspace_t<Backend> &ws) override {
    property_->FillOutput(ws);
  }

 private:
  void PropertyFactory(const std::string &property_key);

  std::unique_ptr<detail::Property<Backend>> property_;
};

}  // namespace dali


#endif  // DALI_OPERATORS_UTIL_GET_PROPERTY_H_
