// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_UTIL_AXES_UTILS_H_
#define DALI_OPERATORS_UTIL_AXES_UTILS_H_

#include "dali/core/small_vector.h"
#include "dali/core/tensor_layout.h"
#include "dali/operators.h"
#include "dali/pipeline/operator/op_spec.h"

namespace dali {

/**
 * @brief Checks that axes only appear once and that they are within range.
 *
 * @param axes list of axis indices
 * @param ndim dimensionality of the tensor(list) to which axes refer
 */
template <typename Axes>
inline void CheckAxes(Axes&& axes, int ndim) {
  assert(ndim >= 0 && ndim <= 64);
  uint64_t mask = 0;
  for (auto a : axes) {
    if (a < 0 || a >= ndim)
      throw std::out_of_range(
          make_string("Axis index out of range: ", a, " not in range [", 0, "..", ndim - 1, "]"));
    uint64_t amask = 1_u64 << a;
    if (mask & amask)
      throw std::invalid_argument(make_string("Duplicate axis index ", a));
    mask |= amask;
  }
}


/**
 * @brief Adjusts negative axis indices to the positive range.
 *        Negative indices are counted from the back.
 *
 * @param axes list of axis indices
 * @param ndim dimensionality of the tensor(list) to which axes refer
 */
template <typename Axes>
void ProcessNegativeAxes(Axes&& axes, int ndim) {
  for (auto& a : axes) {
    if (a < -ndim || a >= ndim)
      throw std::out_of_range(make_string("Axis index out of range: ", a, " not in range [", -ndim,
                                          ",", ndim - 1, "]"));
    if (a < 0)
      a += ndim;
  }
}

class AxesHelper {
 public:
  explicit inline AxesHelper(const OpSpec &spec) {
    has_axes_arg_ = spec.TryGetRepeatedArgument(axes_, "axes");
    has_axis_names_arg_ = spec.TryGetArgument(axis_names_, "axis_names");
    has_empty_axes_arg_ =
      (has_axes_arg_ && axes_.empty()) || (has_axis_names_arg_ && axis_names_.empty());

    DALI_ENFORCE(!has_axes_arg_ || !has_axis_names_arg_,
      "Arguments `axes` and `axis_names` are mutually exclusive");
  }

  inline void PrepareAxes(const TensorLayout &layout, int sample_dim) {
    if (has_axis_names_arg_) {
      axes_ = GetDimIndices(layout, axis_names_).to_vector();
      return;
    }

    if (!has_axes_arg_) {
      axes_.resize(sample_dim);
      std::iota(axes_.begin(), axes_.end(), 0);
    }

    // adjusts negative indices to positive range
    ProcessNegativeAxes(make_span(axes_), sample_dim);
    // checks range and duplicates
    CheckAxes(make_cspan(axes_), sample_dim);
  }

  span<int> Axes() {
    return make_span(axes_);
  }

  bool has_axes_arg_;
  bool has_axis_names_arg_;
  bool has_empty_axes_arg_;
  SmallVector<int, 6> axes_;
  TensorLayout axis_names_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_UTIL_AXES_UTILS_H_
