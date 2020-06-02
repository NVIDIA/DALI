// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_H_
#define DALI_OPERATORS_H_

#ifdef __cplusplus
///@{
namespace dali {
/**
 * @brief Functions to initialize operators in DALI
 *
 * Remember to also call @see daliInitialize() from `c_api.h` to initialize whole DALI backend.
 * Use either InitOperatorsLib or daliInitOperators, depending whether you use C++ or C API.
 * This function initializes operators' library. In case you use only Operators
 * (no Executor or Pipeline), you may pass on @see daliInitialize().
 */
void InitOperatorsLib();
}  // namespace dali
extern "C" void daliInitOperators();
///@}
#else
/**
 * @brief Functions to initialize operators in DALI
 *
 * Remember to also call @see daliInitialize() from `c_api.h` to initialize whole DALI backend.
 * Use either InitOperatorsLib or daliInitOperators, depending whether you use C++ or C API.
 * This function initializes operators' library. In case you use only Operators
 * (no Executor or Pipeline), you may pass on @see daliInitialize().
 */
void daliInitOperators();
#endif


#endif  // DALI_OPERATORS_H_
