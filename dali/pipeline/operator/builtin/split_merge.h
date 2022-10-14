// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_BUILTIN_SPLIT_MERGE_H_
#define DALI_PIPELINE_OPERATOR_BUILTIN_SPLIT_MERGE_H_

#include <string>
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/operator/op_schema.h"

namespace dali {

/**
 * @brief When we have logical condition, the true category is expected to go through the first (0)
 * output and input (respectively for split and merge), so we basically do the reverse here.
 *
 * @param condition
 * @param condition_idx
 * @param is_logical
 * @return int
 */
inline int get_category_index(const TensorList<CPUBackend>& condition, int condition_idx,
                              bool is_logical = true) {
  assert(is_logical && "Numerical conditions are not implemented");
  bool cond_val = *condition.tensor<bool>(condition_idx);
  //
  return cond_val ? 0 : 1;
}

inline bool isSplit(const OpSchema &schema) {
  return schema.name().rfind("_Split") != std::string::npos;
}

inline bool isMerge(const OpSchema &schema) {
  return schema.name().rfind("_Merge") != std::string::npos;
}

inline bool isSplitOrMerge(const OpSchema &schema) {
  return isSplit(schema) || isMerge(schema);
}

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_SPLIT_MERGE_H_
