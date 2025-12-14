// Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_GENERIC_SLICE_OUT_OF_BOUNDS_POLICY_H_
#define DALI_OPERATORS_GENERIC_SLICE_OUT_OF_BOUNDS_POLICY_H_

#include <algorithm>
#include <string>
#include "dali/core/math_util.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/core/boundary.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

/**
 * @brief Determines what to do if slice parameters point to outside of the input bounds
 */
enum class OutOfBoundsShapePolicy {
  Error,        // sampling out of bounds will throw an error
  TrimToShape,  // Slice shape will be trimmed to fit the input bounds (potentially empty output)
  Pad,  // Slicing out of bounds will result in some sort of padding
};

struct OutOfBoundsPolicy {
  OutOfBoundsShapePolicy shape_policy = OutOfBoundsShapePolicy::Error;
  boundary::BoundaryType border_type = boundary::BoundaryType::TRANSPARENT;
};

inline OutOfBoundsPolicy GetOutOfBoundsPolicy(
      const OpSpec &spec,
      std::initializer_list<boundary::BoundaryType> allowed_border_types = {}) {
  bool has_out_of_bounds_policy = spec.HasArgument("out_of_bounds_policy");
  OutOfBoundsPolicy policy{};
  if (has_out_of_bounds_policy) {
    auto policy_str = spec.GetArgument<std::string>("out_of_bounds_policy");
    if (policy_str == "trim_to_shape") {
      policy.shape_policy = OutOfBoundsShapePolicy::TrimToShape;
    } else if (policy_str == "error") {
      policy.shape_policy = OutOfBoundsShapePolicy::Error;
    } else {
      if (!boundary::TryParse(policy.border_type, policy_str)) {
        DALI_FAIL(make_string("Not supported out_of_bounds_policy: ", policy_str));
      }
      bool is_allowed = true;
      if (allowed_border_types.size() > 0) {
        if (std::ranges::none_of(allowed_border_types, [&](boundary::BoundaryType type) {
          return type == policy.border_type;
        })) is_allowed = false;
      } else if (policy.border_type == boundary::BoundaryType::TRANSPARENT) {
        is_allowed = false;
      }
      if (!is_allowed) {
        DALI_FAIL(make_string("Not supported out_of_bounds_policy: ", policy_str));
      }
      policy.shape_policy = OutOfBoundsShapePolicy::Pad;
    }
  }
  return policy;
}

template <int Dims>
void ApplySliceBoundsPolicy(OutOfBoundsShapePolicy policy, const TensorShape<Dims> &input_shape,
                            TensorShape<Dims> &slice_anchor, TensorShape<Dims> &slice_shape) {
  DALI_ENFORCE(
      input_shape.size() == slice_anchor.size() && input_shape.size() == slice_shape.size(),
      "Slice arguments should have the same number of dimensions as the input");
  switch (policy) {
    case OutOfBoundsShapePolicy::Pad:
      // nothing to do
      break;

    case OutOfBoundsShapePolicy::TrimToShape:
      for (int d = 0; d < input_shape.size(); d++) {
        auto slice_start = clamp<int64_t>(slice_anchor[d], 0, input_shape[d]);
        auto slice_end   = clamp<int64_t>(slice_anchor[d] + slice_shape[d], 0, input_shape[d]);
        assert(slice_end >= slice_start);
        slice_anchor[d] = slice_start;
        slice_shape[d] = slice_end - slice_start;
      }
      break;

    case OutOfBoundsShapePolicy::Error:
    default:
      for (int d = 0; d < input_shape.size(); d++) {
        auto range_bound_valid = [](int64_t bound, int64_t data_extent) {
          // Check if the given range bound is valid - unlike index, a bound equal to the extent
          // is valid and may serve as a valid lower bound if the requested range is empty.
          return static_cast<uint64_t>(bound) > static_cast<uint64_t>(data_extent);
        };

        if (range_bound_valid(slice_anchor[d], input_shape[d]) ||
            range_bound_valid(slice_anchor[d] + slice_shape[d], input_shape[d])) {
          DALI_FAIL(make_string(
              "Slice can't be placed out of bounds with current policy. Got: input_shape={",
              input_shape, "}, slice_anchor={", slice_anchor, "}, slice_shape={", slice_shape,
              "}"));
        }
      }
    break;
  }
}

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_SLICE_OUT_OF_BOUNDS_POLICY_H_
