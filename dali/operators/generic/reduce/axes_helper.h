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


#ifndef DALI_OPERATORS_GENERIC_REDUCE_AXIS_HELPER_H__
#define DALI_OPERATORS_GENERIC_REDUCE_AXIS_HELPER_H__

#include <vector>

#include "dali/pipeline/operator/operator.h"

namespace dali {
namespace detail {

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

  void PrepareAxes(const TensorLayout &layout, int sample_dim) {
    if (has_axis_names_arg_) {
      axes_ = GetDimIndices(layout, axis_names_).to_vector();
      return;
    }

    if (!has_axes_arg_) {
      axes_.resize(sample_dim);
      std::iota(axes_.begin(), axes_.end(), 0);
    }
  }

  bool has_axes_arg_;
  bool has_axis_names_arg_;
  bool has_empty_axes_arg_;
  std::vector<int> axes_;
  TensorLayout axis_names_;
};

}  // namespace detail
}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_REDUCE_AXIS_HELPER_H__