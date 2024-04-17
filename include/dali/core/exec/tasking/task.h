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

#ifndef DALI_CORE_EXEC_TASKING_TASK_H_
#define DALI_CORE_EXEC_TASKING_TASK_H_

#include <cassert>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "dali/core/small_vector.h"
#include "dali/core/exec/tasking/sync.h"

namespace dali::tasking {

class Scheduler;


}  // namespace dali::tasking

#endif  // DALI_CORE_EXEC_TASKING_TASK_H_
