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
#include <mutex>
#include <condition_variable>
#include "dali/core/call_at_exit.h"
#include "dali/core/error_handling.h"
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
      throw std::logic_error("This job has already been started - cannot add more tasks to it");
    auto it = tasks_.emplace(priority, Task());
    try {
      it->second.func = [this, task = &it->second, f = std::move(runnable)](int tid) noexcept {
        try {
          f(tid);
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
                             // may go below 0, but we don't care
    }
    if (wait)
      Wait();
  }

  void Wait();

  void Scrap() {
    if (started_)
      throw std::logic_error("Cannot scrap a job that has already been started");
    tasks_.clear();
  }

 private:
  std::mutex mtx_;  // could just probably use atomic_wait on num_pending_tasks_ - needs C++20
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
                mm::detail::object_pool_allocator<std::pair<const priority_t, Task>>> tasks_;
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

  void AddTask(TaskFunc f);

  bool StopRequested() const noexcept {
    return stop_requested_;
  }

  static ThreadPoolBase *this_thread_pool() {
    return this_thread_pool_;
  }

  static int this_thread_idx() {
    return this_thread_idx_;
  }

 protected:
  virtual void OnThreadStart(int thread_idx) noexcept {}
  virtual void OnThreadStop(int thread_idx) noexcept {}

  friend class Job;

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
      while (!tasks_.empty())
        tasks_.pop();

      threads_.clear();
    }
  }

  template <typename Condition>
  bool WaitOrRunTasks(std::condition_variable &cv, Condition &&condition) {
    assert(this_thread_pool() == this);
    std::unique_lock lock(mtx_);
    while (!stop_requested_) {
      bool ret;
      while (!(ret = condition()) && !stop_requested_ && tasks_.empty())
        cv.wait_for(lock, std::chrono::microseconds(100));

      if (ret || condition())  // re-evaluate the condition, just in case
        return true;
      if (stop_requested_)
        return false;
      assert(!tasks_.empty());

      PopAndRunTask(lock);
    }
    return false;
  }

  void PopAndRunTask(std::unique_lock<std::mutex> &mtx);

  static thread_local ThreadPoolBase *this_thread_pool_;
  static thread_local int this_thread_idx_;

  void Run(int index) noexcept;

  std::mutex mtx_;
  std::condition_variable cv_;
  bool stop_requested_ = false;
  std::queue<TaskFunc> tasks_;
  std::vector<std::thread> threads_;
};

//////////////////////////////

void Job::Wait() {
  if (!started_)
    throw std::logic_error("This job hasn't been run - cannot wait for it.");

  if (waited_for_)
    throw std::logic_error("This job has already been waited for.");

  auto ready = [&]() { return num_pending_tasks_ == 0; };

  if (ThreadPoolBase::this_thread_pool() != nullptr) {
    bool result = ThreadPoolBase::this_thread_pool()->WaitOrRunTasks(cv_, ready);
    waited_for_ = true;
    if (!result)
      throw std::runtime_error("The thread pool was stopped");
  } else {
    std::unique_lock lock(mtx_);
    cv_.wait(lock, ready);
    waited_for_ = true;
  }

  // note - this vector is not allocated unless there were exceptions thrown
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



thread_local ThreadPoolBase *ThreadPoolBase::this_thread_pool_ = nullptr;
thread_local int ThreadPoolBase::this_thread_idx_ = -1;;

inline void ThreadPoolBase::AddTask(TaskFunc f) {
  {
    std::lock_guard<std::mutex> g(mtx_);
    if (stop_requested_)
      throw std::logic_error("The thread pool is stopped and no longer accepts new tasks.");
    tasks_.push(std::move(f));
  }
  cv_.notify_one();
}

inline void ThreadPoolBase::Run(int index) noexcept {
  ThreadPoolBase *this_thread_pool_ = this;
  this_thread_idx_ = index;
  OnThreadStart(index);
  detail::CallAtExit([&]() { OnThreadStop(index); });
  std::unique_lock lock(mtx_);
  while (!stop_requested_) {
    cv_.wait(lock, [&]() { return stop_requested_ || !tasks_.empty(); });
    if (stop_requested_)
      break;
    PopAndRunTask(lock);
  }
}

inline void ThreadPoolBase::PopAndRunTask(std::unique_lock<std::mutex> &lock) {
  TaskFunc t = std::move(tasks_.front());
  tasks_.pop();
  lock.unlock();
  t(this_thread_idx());
  lock.lock();
}


}  // namespace experimental
}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_NEW_THREAD_POOL_H_
