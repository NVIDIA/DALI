// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_GENERIC_TRANSPOSE_TRANSPOSE_H_
#define DALI_OPERATORS_GENERIC_TRANSPOSE_TRANSPOSE_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>
#include <utility>

#include "dali/core/permute.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class Transpose : public StatelessOperator<Backend> {
 public:
  explicit inline Transpose(const OpSpec &spec)
      : StatelessOperator<Backend>(spec),
        transpose_layout_(spec.GetArgument<bool>("transpose_layout")),
        output_layout_arg_(spec.GetArgument<TensorLayout>("output_layout")) {
    if (spec.HasArgument("perm"))
      perm_ = spec.GetRepeatedArgument<int>("perm");
    // perm_ is later populated based on the number of dims
    // in case of default permutation
    default_perm_ = perm_.empty();

    if (spec.HasArgument("output_layout")) {
      DALI_ENFORCE(!output_layout_arg_.empty(),
        "Providing an empty output layout is not supported");
    }

    auto check_permutation =
      [](std::vector<int> perm) {
        std::sort(perm.begin(), perm.end());
        for (int i = 0; i < static_cast<int>(perm.size()); ++i) {
          if (perm[i] != i) {
            return false;
          }
        }
        return true;
      };

    DALI_ENFORCE(check_permutation(perm_),
      "Invalid permutation: sorted `perm` is not equal to [0, ..., n-1].");
  }

  DISABLE_COPY_MOVE_ASSIGN(Transpose);

 protected:
  template <typename InputType>
  void SetOutputLayout(const InputType &input) {
    auto in_layout = input.GetLayout();
    auto sample_ndim = input.shape().sample_dim();
    DALI_ENFORCE(in_layout.ndim() == sample_ndim || in_layout.empty());
    output_layout_ = in_layout;
    if (!output_layout_arg_.empty()) {
      DALI_ENFORCE(output_layout_arg_.ndim() == sample_ndim);
      output_layout_ = output_layout_arg_;
    } else if (transpose_layout_ && !in_layout.empty()) {
      output_layout_ = permute(in_layout, perm_);
    }
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc,
                 const Workspace &ws) override {
    const auto &input = ws.Input<Backend>(0);
    const auto &input_shape = input.shape();
    int ndim = input_shape.sample_dim();

    if (default_perm_ && static_cast<int>(perm_.size()) != ndim) {
      perm_.resize(ndim);
      for (int i = 0; i < ndim; i++)
        perm_[i] = ndim - 1 - i;
    }

    SetOutputLayout(input);
    output_desc.resize(1);
    output_desc[0].type = input.type();
    output_desc[0].shape = input_shape;

    if (!perm_.empty()) {
      int dim = input_shape.sample_dim();
      int pdim = perm_.size();
      DALI_ENFORCE(dim == pdim, make_string("Input has different dimensionality (", dim,
          ") than the length of the permutation (", pdim, ")"));
      permute_dims(output_desc[0].shape, input_shape, perm_);
    }
    return true;
  }

  bool CanInferOutputs() const override {
    return true;
  }

 protected:
  bool transpose_layout_;
  TensorLayout output_layout_arg_;
  TensorLayout output_layout_;
  std::vector<int> perm_;
  bool default_perm_ = false;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_TRANSPOSE_TRANSPOSE_H_
