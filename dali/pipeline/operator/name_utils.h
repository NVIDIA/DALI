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

#include "dali/pipeline/operator/op_spec.h"

namespace dali {

/** @brief Enum representing how module path can be formatted for given operator */
enum class ModuleSpecKind {
  /** @brief Only show operator display name */
  OpOnly,
  /** @brief Include the module path specified in schema, for example `experimental.random` */
  Module,
  /** Include the API distinction and module path, for example: `fn.experimental.random` */
  ApiModule,
  /** Include library, api and module path, for example: `nvidia.dali.fn.experimental.random` */
  LibApiModule,
};

/**
 * @brief Get the name of the API this operator was instantiated for.
 */
std::string GetOpApi(const OpSpec &spec);

// /**
//  * @brief Get the module path of the operator
//  */
// std::vector<std::string> GetOpModulePath(const OpSpec &spec, );

/**
 * @brief Get the module path of the operator as a dot separated string.
 *
 * Doesn't include the operator name, ModuleSpecKind::OpOnly is invalid.
 * @param spec OpSpec definition of the operator
 * @param kind Controls which portion of module path should be returned.
 */
std::string GetOpModule(const OpSpec &spec, ModuleSpecKind kind = ModuleSpecKind::Module);


/**
 * @brief Get the display name of the operator, optionally including the module path as a dot
 * separated string.
 *
 * @param spec OpSpec definition of the operator
 * @param kind Controls which portion of module path should be returned.
 */
std::string GetOpDisplayName(const OpSpec &spec, ModuleSpecKind kind = ModuleSpecKind::OpOnly);

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_NAME_UTILS_H_
