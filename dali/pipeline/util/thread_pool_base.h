// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_UTIL_THREAD_POOL_BASE_H_
#define DALI_PIPELINE_UTIL_THREAD_POOL_BASE_H_

#include <cassert>
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <mutex>
#include <condition_variable>
#include "dali/core/api_helper.h"
#include "dali/core/multi_error.h"
#include "dali/core/mm/detail/aux_alloc.h"

namespace dali {
namespace experimental {

class ThreadPoolBase;

/**
 * @brief A collection of tasks, ordered by priority
 *
 * Tasks are added to a job first and then the entire work is scheduled as a whole.
 */
class DLL_PUBLIC Job {
 public:
  ~Job() noexcept(false);

  using priority_t = int64_t;

  template <typename Runnable>
  std::enable_if_t<std::is_convertible_v<Runnable, std::function<void()>>>
  AddTask(Runnable &&runnable, priority_t priority = {}) {
    if (started_)
      throw std::logic_error("This job has already been started - cannot add more tasks to it");
    auto it = tasks_.emplace(priority, Task());
    try {
      it->second.func = [this, task = &it->second, f = std::move(runnable)]() noexcept {
        try {
          f();
        } catch (...) {
          task->error = std::current_exception();
        }
        if (--num_pending_tasks_ == 0) {
          std::lock_guard<std::mutex> g(mtx_);
          cv_.notify_one();
        }
      };
    } catch (...) {  // if, for whatever reason, we cannot initialize the task, we should erase it
      tasks_.erase(it);
      throw;
    }
  }

  template <typename Executor>
  void Run(Executor &executor, bool wait) {
    if (started_)
      throw std::logic_error("This job has already been started.");
    started_ = true;
    for (auto &x : tasks_) {
      executor.AddTask(std::move(x.second.func));
      num_pending_tasks_++;  // increase after successfully scheduling the task - the value
                             // may hit 0 or go below if the task is done before we increment
                             // the counter, but we don't care if we aren't waiting yet
    }
    if (wait)
      Wait();
  }

  void Wait();

  void Scrap();

 private:
  std::mutex mtx_;  // could just probably use atomic_wait on num_pending_tasks_ - needs C++20
  std::condition_variable cv_;
  std::atomic_int num_pending_tasks_{0};
  bool started_ = false;
  bool waited_for_ = false;

  struct Task {
    std::function<void()> func;
    std::exception_ptr error;
  };

  // This needs to be a container which never invalidates references when inserting new items.
  std::multimap<priority_t, Task, std::greater<priority_t>,
                mm::detail::object_pool_allocator<std::pair<const priority_t, Task>>> tasks_;
};

class DLL_PUBLIC ThreadPoolBase {
 public:
  using TaskFunc = std::function<void()>;

  ThreadPoolBase() = default;
  explicit ThreadPoolBase(int num_threads) {
    Init(num_threads);
  }

  void Init(int num_threads);

  ~ThreadPoolBase() {
    Shutdown();
  }

  void AddTask(TaskFunc f);

  /**
   * @brief Returns the thread pool that owns the calling thread (or nullptr)
   */
  static ThreadPoolBase *this_thread_pool() {
    return this_thread_pool_;
  }

  /**
   * @brief Returns the index of the current thread within the current thread pool
   *
   * @return the thread index or -1 if the calling thread does not belong to a thread pool
   */
  static int this_thread_idx() {
    return this_thread_idx_;
  }

 protected:
  void Shutdown();

 private:
  friend class Job;

  virtual void OnThreadStart(int thread_idx) noexcept {}
  virtual void OnThreadStop(int thread_idx) noexcept {}

  template <typename Condition>
  bool WaitOrRunTasks(std::condition_variable &cv, Condition &&condition);

  void PopAndRunTask(std::unique_lock<std::mutex> &mtx);

  static thread_local ThreadPoolBase *this_thread_pool_;
  static thread_local int this_thread_idx_;

  void Run(int index) noexcept;

  std::mutex mtx_;
  std::condition_variable cv_;
  bool shutdown_pending_ = false;
  std::queue<TaskFunc> tasks_;
  std::vector<std::thread> threads_;
};

}  // namespace experimental
}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_THREAD_POOL_BASE_H_
