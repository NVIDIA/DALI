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

#include "dali/kernels/common/utils.h"
#include "dali/operators/generic/transpose/cutt/cutt.h"
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
 *
 * Optionally convert from Row Major to Column Major compatible description (required by cuTT)
 */
template <typename ShapeT = int>
void PrepareArguments(SmallVector<ShapeT, kStaticShapeElements> &shape, VecInt &perm,
                      bool transpose = false) {
  DALI_ENFORCE(shape.size() == perm.size());

  // cuTT does not handle dimensions with size 1 so we remove them
  // (H, W, 1) is equivalent to (H, W)
  auto it_shape = shape.begin();
  auto it_perm = perm.begin();
  SmallVector<ShapeT, kStaticShapeElements> erased;
  while (it_shape != shape.end()) {
    if (*it_shape == 1) {
      erased.push_back(*it_perm);
      it_shape = shape.erase(it_shape);
      it_perm = perm.erase(it_perm);
    } else {
      ++it_shape;
      ++it_perm;
    }
  }
  // when some permutation element is erased all elements positions after it should be decreased
  // by one like it doesn't exist at all
  // sort elements to erase in descending order so we avoid situations like
  // erased(0, 2), perm(3, 1) -> perm(2, 0) while it should be (1, 0)
  std::sort(erased.begin(), erased.end(), std::greater<int>());
  for (auto &pos : erased) {
    for (auto &elm : perm) {
       if (elm > pos) {
         --elm;
       }
    }
  }
  if (transpose) {
    RowToColumnMajor(shape.data(), perm.data(), shape.size());
  }
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
  bool SetupImpl(std::vector<OutputDesc> &output_desc,
                 const workspace_t<Backend> &ws) override {
    const auto &input = ws.template InputRef<Backend>(0);
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
