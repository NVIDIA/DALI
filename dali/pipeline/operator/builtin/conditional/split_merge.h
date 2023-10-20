// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_BUILTIN_CONDITIONAL_SPLIT_MERGE_H_
#define DALI_PIPELINE_OPERATOR_BUILTIN_CONDITIONAL_SPLIT_MERGE_H_

#include <string>
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/builtin/conditional/validation.h"
#include "dali/pipeline/operator/op_schema.h"

namespace dali {

/**
 * @brief When we have logical condition, the true group is expected to go through the first (0)
 * output and input (respectively for split and merge), so we basically do the reverse here.
 *
 * @param condition Argument input describing the split
 * @param condition_idx Index of the sample that this condition applies to (input for split, output
 * for merge)
 */
inline int get_group_index(const TensorList<CPUBackend> &condition, int condition_idx,
                           bool is_logical = true) {
  assert(is_logical && "Numerical conditions are not implemented");
  bool cond_val = {};

  TYPE_SWITCH(condition.type(), type2id, T, LOGICALLY_EVALUATABLE_TYPES, (
    // evaluate as bool here by narrowing conversion
    cond_val = *condition.tensor<T>(condition_idx);
  ), (DALI_FAIL(make_string("Can't evaluate ", condition.type(), " as boolean value."))));  // NOLINT

  return cond_val ? 0 : 1;
}

inline bool IsSplit(const OpSchema &schema) {
  return schema.name() == "_conditional__Split";
}

inline bool IsMerge(const OpSchema &schema) {
  return schema.name() == "_conditional__Merge";
}

inline bool IsSplitOrMerge(const OpSchema &schema) {
  return IsSplit(schema) || IsMerge(schema);
}

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_CONDITIONAL_SPLIT_MERGE_H_
