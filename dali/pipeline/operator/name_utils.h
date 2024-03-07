// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#ifndef DALI_PIPELINE_OPERATOR_NAME_UTILS_H_
#define DALI_PIPELINE_OPERATOR_NAME_UTILS_H_

#include <string>
#include <vector>

#include "dali/core/api_helper.h"
#include "dali/pipeline/operator/op_spec.h"

namespace dali {

/**
 * @brief Get the module path of the operator as a dot separated string.
 *
 * Doesn't include the operator name, fully qualified name is used, like `nvidia.dali.fn.uniform`.
 * @param spec OpSpec definition of the operator
 */
DLL_PUBLIC std::string GetOpModule(const OpSpec &spec);


/**
 * @brief Get the display name of the operator, optionally including the module path as a dot
 * separated string.
 *
 * @param spec OpSpec definition of the operator
 * @param include_module_path if true, the module path is included in the display name
 */
DLL_PUBLIC std::string GetOpDisplayName(const OpSpec &spec, bool include_module_path = false);

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_NAME_UTILS_H_
