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

#ifndef DALI_OPERATORS_GENERIC_TRANSPOSE_TRANSPOSE_H_
#define DALI_OPERATORS_GENERIC_TRANSPOSE_TRANSPOSE_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>
#include <utility>

#include "dali/kernels/common/utils.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

namespace transpose_detail {

template <typename ShapeT>
inline void RowToColumnMajor(ShapeT *dims, int *perm, size_t len) {
  std::reverse(dims, dims + len);
  std::reverse(perm, perm + len);
  for (size_t i = 0; i < len; ++i) perm[i] = len - 1 - perm[i];
}

// enough to represent batches of 4D
constexpr int kStaticShapeElements = 6;
using VecInt = SmallVector<int, kStaticShapeElements>;

/**
 * @brief Remove dimensions equal to 1 and adjust the shape & permutation
 */
template <typename ShapeT = int>
void PrepareArguments(SmallVector<ShapeT, kStaticShapeElements> &shape, VecInt &perm) {
  DALI_ENFORCE(shape.size() == perm.size());
  const int N = shape.size();

  // This will be oterwise reduced to empty shape, and we still want to have some
  // notion of non-empty shape left
  if (volume(shape) == 1) {
    shape = {1};
    perm = {0};
    return;
  }

  SmallVector<ShapeT, kStaticShapeElements> tmp_shape;
  SmallVector<int, kStaticShapeElements> coord_map;
  coord_map.resize(N);

  // Skip all dimensions of shape that are equal to 1. `coord_map` will hold the new "index"
  // for the dimensions that accounts for the skipped ones (valid only for the ones that are left)
  for (int i = 0, skipped = 0; i < N; i++) {
    if (shape[i] == 1) {
      skipped++;
    } else {
      tmp_shape.push_back(shape[i]);
    }
    coord_map[i] = i - skipped;
  }

  VecInt tmp_perm;
  for (int i = 0; i < N; i++) {
    // We need to skip the elements of permutation which correspond to dimensions with extent = 1
    if (shape[perm[i]] == 1) {
      continue;
    }
    // otherwise we pass the element to the new perm and use the new index of this dimension,
    // accounting for the skipped dims
    tmp_perm.push_back(coord_map[perm[i]]);
  }

  perm = std::move(tmp_perm);
  shape = std::move(tmp_shape);
}

}  // namespace transpose_detail


template <typename Backend>
class Transpose : public Operator<Backend> {
 public:
  explicit inline Transpose(const OpSpec &spec)
      : Operator<Backend>(spec),
        perm_(spec.GetRepeatedArgument<int>("perm")),
        transpose_layout_(spec.GetArgument<bool>("transpose_layout")),
        output_layout_arg_(spec.GetArgument<TensorLayout>("output_layout")) {
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
      DALI_ENFORCE(output_layout_.ndim() == sample_ndim);
      output_layout_ = output_layout_arg_;
    } else if (transpose_layout_ && !in_layout.empty()) {
      output_layout_ = kernels::Permute(in_layout, perm_);
    }
  }


  bool SetupImpl(std::vector<OutputDesc> &output_desc,
                 const workspace_t<Backend> &ws) override {
    const auto &input = ws.template InputRef<Backend>(0);
    SetOutputLayout(input);

    output_desc.resize(1);
    if (is_uniform(input.shape())) {
      auto permuted_dims = kernels::Permute(input.shape()[0], perm_);
      output_desc[0].shape = uniform_list_shape(batch_size_, permuted_dims);
    } else {
      TensorListShape<> tl_shape(batch_size_, input.shape().sample_dim());
      for (int i = 0; i < batch_size_; ++i) {
        auto in_shape = input.shape().tensor_shape(i);
        tl_shape.set_tensor_shape(i, kernels::Permute(in_shape, perm_));
      }
      output_desc[0].shape = tl_shape;
    }
    output_desc[0].type = input.type();

    return true;
  }

  bool CanInferOutputs() const override {
    return true;
  }

 protected:
  std::vector<int> perm_;
  bool transpose_layout_;
  TensorLayout output_layout_arg_;
  TensorLayout output_layout_;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_TRANSPOSE_TRANSPOSE_H_
