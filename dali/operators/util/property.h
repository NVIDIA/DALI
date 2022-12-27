// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_UTIL_PROPERTY_H_
#define DALI_OPERATORS_UTIL_PROPERTY_H_

#include <string>
#include "dali/pipeline/data/type_traits.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {
namespace tensor_property {

/**
 * Base class for a property of the Tensor.
 * @tparam Backend Backend of the operator.
 */
template <typename Backend>
struct Property {
  Property() = default;
  virtual ~Property() = default;

  /**
   * @return The shape of the tensor containing the property, based on the input to the operator.
   */
  virtual TensorListShape<> GetShape(const TensorList<Backend>& input) = 0;

  /**
   * @return The type of the tensor containing the property, based on the input to the operator.
   */
  virtual DALIDataType GetType(const TensorList<Backend>& input) = 0;

  /**
   * This function implements filling the output of the operator. Its implementation should
   * be similar to any RunImpl function of the operator.
   */
  virtual void FillOutput(Workspace&) = 0;
};


template <typename Backend>
struct SourceInfo : public Property<Backend> {
  TensorListShape<> GetShape(const TensorList<Backend>& input) override {
    TensorListShape<> ret{static_cast<int>(input.num_samples()), 1};
    for (int i = 0; i < ret.size(); i++) {
      ret.set_tensor_shape(i, {static_cast<int64_t>(GetSourceInfo(input, i).length())});
    }
    return ret;
  }

  DALIDataType GetType(const TensorList<Backend>&) override {
    return DALI_UINT8;
  }

  void FillOutput(Workspace &ws) override;

 private:
  const std::string& GetSourceInfo(const TensorList<Backend>& input, size_t idx) {
    return input.GetMeta(idx).GetSourceInfo();
  }
};


template <typename Backend>
struct Layout : public Property<Backend> {
  TensorListShape<> GetShape(const TensorList<Backend>& input) override {
    // Every tensor in the output has the same number of dimensions
    return uniform_list_shape(input.num_samples(), {GetLayout(input, 0).size()});
  }

  DALIDataType GetType(const TensorList<Backend>&) override {
    return DALI_UINT8;
  }

  void FillOutput(Workspace &ws) override;

 private:
  const TensorLayout& GetLayout(const TensorList<Backend>& input, int idx) {
    return input.GetMeta(idx).GetLayout();
  }
};


}  // namespace tensor_property
}  // namespace dali

#endif  // DALI_OPERATORS_UTIL_PROPERTY_H_
