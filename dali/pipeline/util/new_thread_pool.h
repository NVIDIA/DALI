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

#ifndef DALI_PIPELINE_UTIL_NEW_THREAD_POOL_H_
#define DALI_PIPELINE_UTIL_NEW_THREAD_POOL_H_

#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "dali/core/call_at_exit.h"
#include "dali/core/error_handling.h"
#include "dali/core/multi_error.h"
#include "dali/core/mm/detail/aux_alloc.h"

namespace dali {
namespace experimental {

/**
 * @brief A collection of tasks, ordered by priority
 *
 * Tasks are added to a job first and then the entire work is scheduled as a whole.
 */
class Job {
 public:
  ~Job() noexcept(false) {
    if (!tasks_.empty() && !waited_for_)  {
      std::lock_guard<std::mutex> g(mtx_);
      if (!tasks_.empty() && !waited_for_) {
        throw std::logic_error("The job is not empty, but hasn't been scrapped or waited for.");
      }
    }
  }

  using priority_t = int64_t;

  template <typename Runnable>
  std::enable_if_t<std::is_convertible_v<Runnable, std::function<void(int)>>>
  AddTask(Runnable &&runnable, priority_t priority = {}) {
    if (started_)
      throw std::logic_error("This has already been started - cannot add more tasks to it");
    auto it = tasks_.emplace(priority, Task());
    try {
      it->second.func = [this, task = &it->second, f = std::move(runnable)](int tid) {
        try {
          f(tid);
        } catch (...) {
          task->error = std::current_exception();
        }
        if (--num_pending_tasks_ == 0)
          cv_.notify_one();
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
                             // may go below 0, but we don't care
    }
    if (wait)
      Wait();
  }

  void Wait() {
    if (!started_)
      throw std::logic_error("This job hasn't been run - cannot wait for it.");
    {
      std::unique_lock lock(mtx_);
      cv_.wait(lock, [&]() { return num_pending_tasks_ == 0; });
      waited_for_ = true;
    }
    std::vector<std::exception_ptr> errors;
    for (auto &x : tasks_) {
      if (x.second.error)
        errors.push_back(std::move(x.second.error));
    }
    if (errors.size() == 1)
      std::rethrow_exception(errors[0]);
    else if (errors.size() > 1)
      throw MultipleErrors(std::move(errors));
  }

  void Scrap() {
    if (started_)
      throw std::logic_error("Cannot scrap a job that has already been started");
    tasks_.clear();
  }

 private:
  std::mutex mtx_;  // this is a dummy mutex - we could just use atomic_wait on num_pending_tasks_
  std::condition_variable cv_;
  std::atomic_int num_pending_tasks_{0};
  bool started_ = false;
  bool waited_for_ = false;

  struct Task {
    std::function<void(int)> func;
    std::exception_ptr error;
  };

  // This needs to be a container which never invalidates references when inserting new items.
  std::multimap<priority_t, Task, std::greater<priority_t>,
                mm::detail::object_pool_allocator<std::pair<priority_t, Task>>> tasks_;
};

class ThreadPoolBase {
 public:
  using TaskFunc = std::function<void(int)>;

  ThreadPoolBase() = default;
  explicit ThreadPoolBase(int num_threads) {
    Init(num_threads);
  }

  void Init(int num_threads) {
    std::lock_guard<std::mutex> g(mtx_);
    if (!threads_.empty())
      throw std::logic_error("The thread pool is already started!");
    stop_requested_ = false;
    threads_.reserve(num_threads);
    for (int i = 0; i < num_threads; i++)
      threads_.push_back(std::thread(&ThreadPoolBase::Run, this, i));
  }

  ~ThreadPoolBase() {
    Stop();
  }

  void AddTask(TaskFunc f) {
    {
      std::lock_guard<std::mutex> g(mtx_);
      if (stop_requested_)
        throw std::logic_error("The thread pool is stopped and no longer accepts new tasks.");
      tasks_.push(std::move(f));
    }
    cv_.notify_one();
  }

 protected:
  virtual void OnThreadStart(int thread_idx) {}
  virtual void OnThreadStop(int thread_idx) {}

  void Stop() {
    {
      std::lock_guard<std::mutex> g(mtx_);
      stop_requested_ = true;
      cv_.notify_all();
    }

    for (auto &t : threads_)
      t.join();

    {
      std::lock_guard<std::mutex> g(mtx_);
      threads_.clear();
    }
  }

  void Run(int index) noexcept {
    OnThreadStart(index);
    detail::CallAtExit([&]() { OnThreadStop(index); });
    std::unique_lock lock(mtx_);
    while (!stop_requested_) {
      cv_.wait(lock, [&]() { return stop_requested_ || !tasks_.empty(); });
      if (stop_requested_)
        break;
      TaskFunc t = std::move(tasks_.front());
      tasks_.pop();
      lock.unlock();
      t(index);
      lock.lock();
    }
  }

  std::mutex mtx_;
  std::condition_variable cv_;
  bool stop_requested_ = false;
  std::queue<TaskFunc> tasks_;
  std::vector<std::thread> threads_;
};

}  // namespace experimental
}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_NEW_THREAD_POOL_H_
