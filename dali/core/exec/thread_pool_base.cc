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

#include "dali/core/exec/thread_pool_base.h"
#include <stdexcept>
#include <thread>

namespace dali {

template <bool cooperative>
JobBase<cooperative>::~JobBase() noexcept(false) {
  if (total_tasks_ > 0 && !wait_completed_) {
    throw std::logic_error("The job is not empty, but hasn't been discarded or waited for.");
  }
  while (running_)
    std::this_thread::yield();
}

template <bool cooperative>
void JobBase<cooperative>::DoWait() {
  if (wait_started_)
    throw std::logic_error("This job has already been waited for.");
  wait_started_ = true;

  if (total_tasks_ == 0) {
    wait_completed_ = true;
    return;
  }

  if (this->executor_ == nullptr)
    throw std::logic_error("This job hasn't been run - cannot wait for it.");

  if (ThreadPoolBase::this_thread_pool() != nullptr) {
    if constexpr (cooperative) {
      auto ready = [&]() { return num_pending_tasks_ == 0; };
      bool result = ThreadPoolBase::this_thread_pool()->WaitOrRunTasks(this->cv_, ready);
      wait_completed_ = true;
      if (!result)
        throw std::logic_error("The thread pool was stopped");
    } else {
      throw std::logic_error("Cannot wait for this job from inside the thread pool.");
    }
  } else {
    int old = num_pending_tasks_.load();
    while (old != 0) {
      num_pending_tasks_.wait(old);
      old = num_pending_tasks_.load();
      assert(old >= 0);
    }
    wait_completed_ = true;
  }
}

template <bool cooperative>
void JobBase<cooperative>::DoNotify() {
  num_pending_tasks_.notify_all();
  if constexpr (cooperative) {
    (void)std::lock_guard(this->mtx_);
    this->cv_.notify_all();
  }
  // We need this second flag to avoid a race condition where the destructor is called between
  // decrementing num_pending_tasks_ and notification_ without excessive use of mutexes.
  // This must be the very last operation in the task function that touches `this`.
  running_ = false;
}

// Job ////////////////////////////////////////////////////////////////////

template <bool cooperative>
void JobImpl<cooperative>::Run(ThreadPoolBase &tp, bool wait) {
  if (this->executor_ != nullptr)
    throw std::logic_error("This job has already been started.");
  this->executor_ = &tp;
  this->running_ = !tasks_.empty();
  {
    auto batch = tp.BeginBulkAdd();
    for (auto &x : tasks_) {
      batch.Add(std::move(x.second.func));
    }
    int added = batch.Size();
    if (added) {
      this->num_pending_tasks_ += added;
      this->running_ = true;
    }
    batch.Submit();
  }
  if (wait && !tasks_.empty())
    Wait();
}

template <bool cooperative>
void JobImpl<cooperative>::Wait() {
  this->DoWait();

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

template <bool cooperative>
void JobImpl<cooperative>::Discard() {
  if (this->executor_ != nullptr)
    throw std::logic_error("Cannot discard a job that has already been started");
  tasks_.clear();
  this->total_tasks_ = 0;
}

// IncrementalJob /////////////////////////////////////////////////////////

template <bool cooperative>
void IncrementalJobImpl<cooperative>::Run(ThreadPoolBase &tp, bool wait) {
  if (this->executor_ && this->executor_ != &tp)
    throw std::logic_error("This job is already running in a different executor.");
  this->executor_ = &tp;
  {
    auto it = last_task_run_.has_value() ? std::next(*last_task_run_) : tasks_.begin();
    auto batch = tp.BeginBulkAdd();
    for (; it != tasks_.end(); ++it) {
      batch.Add(std::move(it->func));
      last_task_run_ = it;
    }
    int added = batch.Size();
    if (added) {
      this->num_pending_tasks_ += added;
      this->running_ = true;
    }
    batch.Submit();
  }
  if (wait && !tasks_.empty())
    Wait();
}

template <bool cooperative>
void IncrementalJobImpl<cooperative>::Discard() {
  if (this->executor_)
    throw std::logic_error("Cannot discard a job that has already been started");
  tasks_.clear();
  this->total_tasks_ = 0;
}

template <bool cooperative>
void IncrementalJobImpl<cooperative>::Wait() {
  this->DoWait();
  // note - this vector is not allocated unless there were exceptions thrown
  std::vector<std::exception_ptr> errors;
  for (auto &x : tasks_) {
    if (x.error)
      errors.push_back(std::move(x.error));
  }
  if (errors.size() == 1)
    std::rethrow_exception(errors[0]);
  else if (errors.size() > 1)
    throw MultipleErrors(std::move(errors));
}

///////////////////////////////////////////////////////////////////////////

thread_local ThreadPoolBase *ThreadPoolBase::this_thread_pool_ = nullptr;
thread_local int ThisThreadIdx::this_thread_idx_ = -1;

void ThreadPoolBase::Init(int num_threads, const std::function<OnThreadStartFn> &on_thread_start) {
  if (shutdown_pending_)
    throw std::logic_error("The thread pool is being shut down.");
  std::lock_guard<std::mutex> g(mtx_);
  if (!threads_.empty())
    throw std::logic_error("The thread pool is already started!");
  threads_.reserve(num_threads);
  for (int i = 0; i < num_threads; i++)
    threads_.push_back(std::thread(&ThreadPoolBase::Run, this, i, on_thread_start));
}

void ThreadPoolBase::Shutdown(bool join) {
  if ((shutdown_pending_ && !join) || threads_.empty())
    return;
  {
    std::lock_guard<std::mutex> g(mtx_);
    if (shutdown_pending_ && !join)
      return;
    shutdown_pending_ = true;
    sem_.release(threads_.size());
  }

  for (auto &t : threads_)
    t.join();
  threads_.clear();
}

void ThreadPoolBase::AddTaskNoLock(TaskFunc &&f) {
  if (shutdown_pending_)
    throw std::logic_error("The thread pool is stopped and no longer accepts new tasks.");
  tasks_.push(std::move(f));
}

void ThreadPoolBase::AddTask(TaskFunc &&f) {
  {
    std::lock_guard<std::mutex> g(mtx_);
    AddTaskNoLock(std::move(f));
  }
  sem_.release(1);
}

void ThreadPoolBase::Run(
      int index,
      const std::function<OnThreadStartFn> &on_thread_start) noexcept {
  this_thread_pool_ = this;
  this_thread_idx_ = index;
  std::any scope;
  if (on_thread_start)
    scope = on_thread_start(index);
  while (!shutdown_pending_ || !tasks_.empty()) {
    sem_.acquire();
    std::unique_lock lock(mtx_);
    if (shutdown_pending_)
      break;
    assert(!tasks_.empty() && "Semaphore acquired but no tasks present.");
    PopAndRunTask(lock);
  }
}

void ThreadPoolBase::PopAndRunTask(std::unique_lock<std::mutex> &lock) {
  TaskFunc t = std::move(tasks_.front());
  tasks_.pop();
  lock.unlock();
  t();
  lock.lock();
}

template <typename Condition>
bool ThreadPoolBase::WaitOrRunTasks(std::condition_variable &cv, Condition &&condition) {
  assert(this_thread_pool() == this);
  std::unique_lock lock(mtx_);
  while (!shutdown_pending_ || !tasks_.empty()) {
    bool ret;
    while (!(ret = condition()) && tasks_.empty())
      cv.wait_for(lock, std::chrono::microseconds(100));

    if (ret || condition())  // re-evaluate the condition, just in case
      return true;
    if (shutdown_pending_)
      return condition();
    if (!sem_.try_acquire())
      continue;

    assert(!tasks_.empty() && "Semaphore acquired but no tasks present.");
    PopAndRunTask(lock);
  }
  return condition();
}

template class DLL_PUBLIC JobBase<true>;
template class DLL_PUBLIC JobBase<false>;
template class DLL_PUBLIC JobImpl<true>;
template class DLL_PUBLIC JobImpl<false>;
template class DLL_PUBLIC IncrementalJobImpl<true>;
template class DLL_PUBLIC IncrementalJobImpl<false>;

}  // namespace dali
