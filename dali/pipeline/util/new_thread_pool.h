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

#ifndef DALI_PIPELINE_UTIL_THREAD_POOL_H_
#define DALI_PIPELINE_UTIL_THREAD_POOL_H_

#include <memory>
#include <queue>
#include "dali/core/error_handling.h"
#include <map>
#include "dali/core/mm/detail/aux_alloc.h"

namespace dali {
namespace experimental {

class Job {
 public:
  using priority_t = int64_t;

  template <typename Runnable>
  std::enable_if_t<std::is_convertible_v<Runnable, std::function<void(int)>>
  AddTask(Runnable &&runnable, priority_t priority = {}) {
    auto it = tasks.emplace(priority, Task());
    it->second.func = [this, task = &it->second, f = std::move(runnable)](int tid) {
      try {
        f(tid);
      } catch (...) {
        task->error = std::current_exception();
      }
    };
  }

 private:
  struct Task {
    std::function<void(int)> func;
    std::exception_ptr error;
  };

  std::multimap<priority_t, Task, std::greater<priority_t>,
                mm::detail::object_pool_allocator<std::pair<priority_t, Task>>> tasks;
};

class ThreadPool {
 public:
  void ExecuteWork(Work &work);
 private:
};

}  // namespace experimental
}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_THREAD_POOL_H_
