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

#include <string>

#include "dali/pipeline/data/tensor_list.h"

namespace dali {

/**
 * @brief Check if the inputs to logical operation are scalars (and optionally booleans).
 *
 * Report appropriate error with suggested alternatives.
 * @param input The input batch to validate.
 * @param name Name of the operator that we are checking for.
 * @param side Which side of the operator the input came from or empty string if there is only one.
 * @param enforce_type Whether to enforce the type.
 */
void EnforceConditionalInputKind(const TensorList<CPUBackend> &input, const std::string &name,
                                 const std::string &side = "", bool enforce_type = true);

}