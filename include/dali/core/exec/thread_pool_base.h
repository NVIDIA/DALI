// Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_EXEC_THREAD_POOL_BASE_H_
#define DALI_CORE_EXEC_THREAD_POOL_BASE_H_

#include <any>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include "dali/core/api_helper.h"
#include "dali/core/format.h"
#include "dali/core/multi_error.h"
#include "dali/core/semaphore.h"
#include "dali/core/exec/thread_idx.h"
#include "dali/core/mm/detail/aux_alloc.h"

namespace dali {

class ThreadPoolBase;

template <bool cooperative>
class JobBaseFields {};

template <>
class DLL_PUBLIC JobBaseFields<true> {
 protected:
  // atomic wait has no timeout, so we're stuck with condvar for reentrance
  std::mutex mtx_;
  std::condition_variable cv_;
};


/** A base class for various job types. It defines common infrastructure. */
template <bool cooperative>
class DLL_PUBLIC JobBase : public JobBaseFields<cooperative> {
 protected:
  JobBase() = default;
  ~JobBase() noexcept(false);

  /** Waits for all tasks to complete. Errors are NOT rethrown.
   *
   * NOTE: This function must not be inline and must be defined in the same dynamic shared object
   *       as the DoNotify function.
   */
  void DoWait();

  /** Notifies the job that all pending tasks have completed
   *
   * NOTE: This function must not be inline and must be defined in the same dynamic shared object
   *       as the DoWait function.
   */
  void DoNotify();

  static constexpr bool IsCooperative() { return cooperative; }

  std::atomic_int num_pending_tasks_{0};
  std::atomic_bool running_{false};
  int total_tasks_ = 0;
  bool wait_started_ = false;
  bool wait_completed_ = false;
  const void *executor_ = nullptr;

  struct Task {
    std::function<void()> func;
    std::exception_ptr error;
  };
};

/**
 * @brief A collection of tasks, ordered by priority
 *
 * Tasks are added to a job first and then the entire work is scheduled as a whole.
 * Once at least one task has been added, Run and Wait (or Discard) must be called
 * before the task is destroyed.
 */
template <bool cooperative = true>
class DLL_PUBLIC JobImpl final : public JobBase<cooperative> {
 public:
  ~JobImpl() noexcept(false) = default;

  using priority_t = int64_t;
  using Task = JobBase<cooperative>::Task;

  template <typename Runnable>
  std::enable_if_t<std::is_convertible_v<Runnable, std::function<void()>>>
  AddTask(Runnable &&runnable, priority_t priority = {}) {
    if (this->wait_started_)
      throw std::logic_error("This job has already been waited for - cannot add more tasks to it");

    if (this->executor_ != nullptr)
      throw std::logic_error("This job has already been started - cannot add more tasks to it");

    auto it = tasks_.emplace(priority, Task());
    try {
      it->second.func = [this, task = &it->second, f = std::move(runnable)]() noexcept {
        try {
          f();
        } catch (...) {
          task->error = std::current_exception();
        }
        if (--this->num_pending_tasks_ == 0)
          this->DoNotify();
      };
      this->total_tasks_++;
    } catch (...) {  // if, for whatever reason, we cannot initialize the task, we should erase it
      tasks_.erase(it);
      throw;
    }
  }

  template <typename Executor>
  void Run(Executor &executor, bool wait);

  void Run(ThreadPoolBase &tp, bool wait);

  /** Waits for the job to complete. This function must be called only once. */
  void Wait();

  void Discard();

 private:
  // This needs to be a container which never invalidates references when inserting new items.
  std::multimap<priority_t, Task, std::greater<priority_t>,
                mm::detail::object_pool_allocator<std::pair<const priority_t, Task>>> tasks_;
};

/** A job which can be extended with new tasks while already running.
 *
 * Unlike the regular `Job`, this job class doesn't prohibit adding new tasks after
 * calling `Run`. It's still illegal to add new jobs while already waiting for completion.
 *
 * In this job, the tasks are processed strictly in FIFO order - there are no priorities.
 *
 * Calls to AddTask, Run and Wait are not thread safe and require external synchronization if
 * called from different threads.
 */
template <bool cooperative = true>
class DLL_PUBLIC IncrementalJobImpl final : public JobBase<cooperative> {
 public:
  ~IncrementalJobImpl() noexcept(false) = default;

  using Task = JobBase<cooperative>::Task;

  template <typename Runnable>
  std::enable_if_t<std::is_convertible_v<Runnable, std::function<void()>>>
  AddTask(Runnable &&runnable);

  template <typename Executor>
  void Run(Executor &executor, bool wait);

  void Run(ThreadPoolBase &tp, bool wait);

  /** Waits for the job to complete. This function must be called only once.
   *
   * After this call, adding more tasks is illegal.
   */
  void Wait();

  void Discard();

 private:
  using task_list_t = std::list<Task, mm::detail::object_pool_allocator<Task>>;
  task_list_t tasks_;
  std::optional<typename task_list_t::iterator> last_task_run_;
};

using Job = JobImpl<false>;
using CooperativeJob = JobImpl<true>;

using IncrementalJob = IncrementalJobImpl<false>;
using CooperativeIncrementalJob = IncrementalJobImpl<true>;

class DLL_PUBLIC ThreadPoolBase : public ThisThreadIdx {
 public:
  using TaskFunc = std::function<void()>;

  ThreadPoolBase() = default;
  explicit ThreadPoolBase(int num_threads) {
    Init(num_threads);
  }

  /** A function called upon thread start.
   *
   * @param thread_idx Index of the thread within this thread pool.
   * @return A RAII object that lives until the thread's processing loop runs
   *
   * @note This callback doesn't explicitly take `this` pointer - if necessary, a lambda function
   *       can be used that captures the current thread pool instance.
   */
  using OnThreadStartFn = std::any(int thread_idx);

  virtual void Init(int num_threads, const std::function<OnThreadStartFn> &on_thread_start = {});

  virtual ~ThreadPoolBase() {
    Shutdown(true);
  }

  void AddTask(TaskFunc &&f);

  void AddTaskNoLock(TaskFunc &&f);

  class TaskBulkAdd {
   public:
    void Add(TaskFunc &&f) {
      if (!lock.owns_lock())
        lock.lock();
      owner->AddTaskNoLock(std::move(f));
      tasks_added++;
    }

    ~TaskBulkAdd() {
      Submit();
    }

    void Submit() {
      if (lock.owns_lock()) {
        lock.unlock();
        owner->sem_.release(tasks_added);
      }
    }

    int Size() const {
      return tasks_added;
    }

   private:
    friend class ThreadPoolBase;
    explicit TaskBulkAdd(ThreadPoolBase *o) : owner(o), lock(o->mtx_, std::defer_lock) {}
    ThreadPoolBase *owner = nullptr;
    std::unique_lock<std::mutex> lock;
    int tasks_added = 0;
  };
  friend class TaskBulkAdd;

  TaskBulkAdd BeginBulkAdd() & { return TaskBulkAdd(this); }

  int NumThreads() const {
    return threads_.size();
  }

  auto GetThreadIds() const {
    int n = threads_.size();
    std::vector<std::thread::id> ids(n);
    for (int i = 0; i < n; i++)
      ids[i] = threads_[i].get_id();
    return ids;
  }

  /**
   * @brief Returns the thread pool that owns the calling thread (or nullptr)
   */
  static ThreadPoolBase *this_thread_pool() {
    return this_thread_pool_;
  }

 protected:
  void Shutdown(bool join);

 private:
  template <bool cooperative>
  friend class JobBase;

  template <typename Condition>
  bool WaitOrRunTasks(std::condition_variable &cv, Condition &&condition);

  void PopAndRunTask(std::unique_lock<std::mutex> &mtx);

  static thread_local ThreadPoolBase *this_thread_pool_;

  void Run(int index, const std::function<OnThreadStartFn> &on_thread_start) noexcept;

  std::mutex mtx_;
  counting_semaphore sem_{0};
  bool shutdown_pending_ = false;
  std::queue<TaskFunc> tasks_;
  std::vector<std::thread> threads_;
};


template <typename ThreadPool>
class ThreadedExecutionEngine {
 public:
  ThreadedExecutionEngine(ThreadPool &tp) : tp_(tp) {}  // NOLINT

  template <typename FunctionLike>
  void AddWork(FunctionLike &&f, int64_t priority = 0) {
    job_.AddTask(std::forward<FunctionLike>(f), priority);
  }

  void RunAll() {
    job_.Run(tp_, true);
  }

  int NumThreads() const noexcept {
    return tp_.NumThreads();
  }

  ThreadPool &GetThreadPool() const noexcept {
    return tp_;
  }

 private:
  ThreadPool &tp_;
  Job job_;
};

template <bool cooperative>
template <typename Executor>
void JobImpl<cooperative>::Run(Executor &executor, bool wait) {
  if constexpr (std::is_base_of_v<ThreadPoolBase, Executor>) {
    Run(static_cast<ThreadPoolBase &>(executor), wait);
  } else {
    if (this->executor_ != nullptr)
      throw std::logic_error("This job has already been started.");
    this->executor_ = &executor;
    this->running_ = !tasks_.empty();
    for (auto &x : tasks_) {
      this->num_pending_tasks_++;
      try {
        executor.AddTask(std::move(x.second.func));
      } catch (...) {
        if (--this->num_pending_tasks_ == 0)
          this->DoNotify();
        throw;
      }
    }
    if (wait && !tasks_.empty())
      Wait();
  }
}

template <bool cooperative>
template <typename Runnable>
std::enable_if_t<std::is_convertible_v<Runnable, std::function<void()>>>
IncrementalJobImpl<cooperative>::AddTask(Runnable &&runnable) {
  if (this->wait_started_)
    throw std::logic_error("This job has already been waited for - cannot add more tasks to it");

  assert(this->executor_ == nullptr || this->executor_ != ThreadPoolBase::this_thread_pool());

  auto it = tasks_.emplace(tasks_.end(), Task());
  try {
    it->func = [this, task = &*it, f = std::move(runnable)]() noexcept {
      try {
        f();
      } catch (...) {
        task->error = std::current_exception();
      }

      if (--this->num_pending_tasks_ == 0)
        this->DoNotify();
    };
    this->total_tasks_++;
  } catch (...) {  // if, for whatever reason, we cannot initialize the task, we should erase it
    tasks_.erase(it);
    throw;
  }
}

template <bool cooperative>
template <typename Executor>
void IncrementalJobImpl<cooperative>::Run(Executor &executor, bool wait) {
  if constexpr (std::is_base_of_v<ThreadPoolBase, Executor>) {
    Run(static_cast<ThreadPoolBase &>(executor), wait);
  } else {
    if (this->executor_ && this->executor_ != &executor)
      throw std::logic_error("This job is already running in a different executor.");
    this->executor_ = &executor;
    auto it = last_task_run_.has_value() ? std::next(*last_task_run_) : tasks_.begin();
    for (; it != tasks_.end(); ++it) {
      this->running_ = true;
      this->num_pending_tasks_++;
      executor.AddTask(std::move(it->func));
      last_task_run_ = it;
    }
    if (wait && !tasks_.empty())
      Wait();
  }
}

}  // namespace dali

#endif  // DALI_CORE_EXEC_THREAD_POOL_BASE_H_
