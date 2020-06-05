// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

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
void ProcessSliceArgs(OutOfBoundsPolicy policy, const TensorShape<Dims> &input_shape,
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
        if (slice_anchor[d] < 0)
          slice_anchor[d] = 0;
        if (slice_anchor[d] + slice_shape[d] >= input_shape[d])
          slice_shape[d] = std::max(0l, input_shape[d] - slice_anchor[d]);
      }
      break;

    case OutOfBoundsPolicy::Error:
    default:
      for (int d = 0; d < input_shape.size(); d++) {
        auto slice_end = slice_anchor[d] + slice_shape[d];
        if (slice_anchor[d] < 0 || slice_anchor[d] > input_shape[d] || slice_end < 0 ||
            slice_end > input_shape[d]) {
          DALI_FAIL(make_string(
              "Slice can't be place out of bounds with current policy. Got: input_shape={",
              input_shape, "}, slice_anchor={", slice_anchor, "}, slice_shape={", slice_shape,
              "}"));
        }
      }

    break;
  }
}

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_SLICE_OUT_OF_BOUNDS_POLICY_H_
