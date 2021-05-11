// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_INIT_H_
#define DALI_PIPELINE_INIT_H_

#include "dali/core/api_helper.h"

namespace dali {

class OpSpec;

/**
 * @brief Initializes the pipeline. Sets global cpu&gpu allocators for all
 * pipeline objects. Must be called prior to constructing pipeline objects.
 * This must be called only once within a process.
 */
DLL_PUBLIC void DALIInit(const OpSpec &cpu_allocator,
              const OpSpec &pinned_cpu_allocator,
              const OpSpec &gpu_allocator);

}  // namespace dali

#endif  // DALI_PIPELINE_INIT_H_
