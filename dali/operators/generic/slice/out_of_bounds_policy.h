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

#ifndef DALI_OPERATORS_GENERIC_SLICE_OUT_OF_BOUNDS_POLICY_H_
#define DALI_OPERATORS_GENERIC_SLICE_OUT_OF_BOUNDS_POLICY_H_

#include <string>
#include "dali/core/math_util.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

template <bool inclusive_end>
DALI_HOST_DEV DALI_FORCEINLINE bool is_out_of_bounds(int64_t idx, int64_t data_extent) {
  if (inclusive_end)  // check idx is within [0, data_extent]
    return static_cast<uint64_t>(idx) > static_cast<uint64_t>(data_extent);
  else                // check idx is within [0, data_extent)
    return static_cast<uint64_t>(idx) >= static_cast<uint64_t>(data_extent);
}

/**
 * @brief Determines what to do if slice parameters point to outside of the input bounds
 */
enum class OutOfBoundsPolicy {
  Error,        // sampling out of bounds will throw an error
  TrimToShape,  // Slice shape will be trimmed to fit the input bounds (potentially empty output)
  Pad,  // Slicing out of bounds will result in padding with zeroes or any other provided value(s)
};

inline OutOfBoundsPolicy GetOutOfBoundsPolicy(const OpSpec &spec) {
  bool has_out_of_bounds_policy = spec.HasArgument("out_of_bounds_policy");
  OutOfBoundsPolicy policy = OutOfBoundsPolicy::Error;
  if (has_out_of_bounds_policy) {
    auto policy_str = spec.GetArgument<std::string>("out_of_bounds_policy");
    if (policy_str == "pad") {
      policy = OutOfBoundsPolicy::Pad;
    } else if (policy_str == "trim_to_shape") {
      policy = OutOfBoundsPolicy::TrimToShape;
    } else if (policy_str == "error") {
      policy = OutOfBoundsPolicy::Error;
    } else {
      DALI_FAIL(
          make_string("Not supported out_of_bounds_policy: ", policy_str,
                      ". Supported values are \"pad\", \"trim_to_shape\", \"error\" (default)"));
    }
  }
  return policy;
}

template <int Dims>
void ApplySliceBoundsPolicy(OutOfBoundsPolicy policy, const TensorShape<Dims> &input_shape,
                            TensorShape<Dims> &slice_anchor, TensorShape<Dims> &slice_shape) {
  DALI_ENFORCE(
      input_shape.size() == slice_anchor.size() && input_shape.size() == slice_shape.size(),
      "Slice arguments should have the same number of dimensions as the input");
  switch (policy) {
    case OutOfBoundsPolicy::Pad:
      // nothing to do
      break;

    case OutOfBoundsPolicy::TrimToShape:
      for (int d = 0; d < input_shape.size(); d++) {
        auto slice_start = clamp<int64_t>(slice_anchor[d], 0, input_shape[d]);
        auto slice_end   = clamp<int64_t>(slice_anchor[d] + slice_shape[d], 0, input_shape[d]);
        assert(slice_end >= slice_start);
        slice_anchor[d] = slice_start;
        slice_shape[d] = slice_end - slice_start;
      }
      break;

    case OutOfBoundsPolicy::Error:
    default:
      for (int d = 0; d < input_shape.size(); d++) {
        // start within [0, extent), and end within [0, extent]
        if (is_out_of_bounds<false>(slice_anchor[d], input_shape[d]) ||
            is_out_of_bounds<true>(slice_anchor[d] + slice_shape[d], input_shape[d])) {
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
