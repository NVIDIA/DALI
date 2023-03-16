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

#ifndef DALI_PIPELINE_OPERATOR_BUILTIN_CONDITIONAL_VALIDATION_H_
#define DALI_PIPELINE_OPERATOR_BUILTIN_CONDITIONAL_VALIDATION_H_

#include <string>

#include "dali/pipeline/data/tensor_list.h"

namespace dali {

// Types that can be used as input to not expression or in a predicate and be evaluated to boolean.
#define LOGICALLY_EVALUATABLE_TYPES                                                                \
  (bool, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float16, float, \
  double)


/**
 * @brief Check if the inputs to logical expression are scalars (and optionally booleans).
 *
 * Report appropriate error with suggested alternatives.
 * @param input The input batch to validate.
 * @param name Name of the expression that we are checking for.
 * @param where The source of the error, one of "left", "right".
 * @param enforce_type Whether to enforce the boolean type.
 */
void EnforceConditionalInputKind(const TensorList<CPUBackend> &input, const std::string &name,
                                 const std::string &where = "", bool enforce_type = true);

/**
 * @brief Report the error for GPU data in logical expression or if statement.
 * @param name Name of the expression or statement that we are checking for.
 * @param where The source of the error, one of "left", "right" or "if-stmt"
 * @param enforce_type Whether to enforce the boolean type.
 */
void ReportGpuInputError(const std::string &name, const std::string &where = "",
                         bool enforce_type = true);

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_CONDITIONAL_VALIDATION_H_
