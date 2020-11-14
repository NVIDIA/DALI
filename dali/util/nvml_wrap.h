/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ************************************************************************/


// Dynamically handle dependencies on external libraries (other than cudart).

#ifndef DALI_UTIL_NVML_WRAP_H_
#define DALI_UTIL_NVML_WRAP_H_


#include <nvml.h>
#include <cuda_runtime_api.h>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"

bool nvmlIsInitialized(void);
nvmlReturn_t nvmlInitChecked(void);
bool nvmlIsSymbolAvailable(const char *name);

/**
 * Checks, whether CUDA11-proper NVML functions have been successfully loaded
 */
bool nvmlHasCuda11NvmlFunctions(void);

#endif  // DALI_UTIL_NVML_WRAP_H_

