// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_EXEC_THREAD_IDX_H_
#define DALI_CORE_EXEC_THREAD_IDX_H_

#include "dali/core/api_helper.h"

namespace dali {

class DLL_PUBLIC ThisThreadIdx {
 public:
  /**
   * @brief Returns the index of the current thread within the current thread pool
   *
   * @return the thread index or -1 if the calling thread does not belong to a thread pool
   */
  static int this_thread_idx() {
    return this_thread_idx_;
  }

 protected:
  static thread_local int this_thread_idx_;
};

}  // namespace dali

#endif  // DALI_CORE_EXEC_THREAD_IDX_H_
