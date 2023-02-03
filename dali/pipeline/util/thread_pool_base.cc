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

#include "dali/pipeline/util/thread_pool_base.h"
#include "dali/core/call_at_exit.h"

namespace dali {
namespace experimental {

Job::~Job() noexcept(false) {
  if (!tasks_.empty() && !waited_for_)  {
    std::lock_guard<std::mutex> g(mtx_);
    if (!tasks_.empty() && !waited_for_) {
      throw std::logic_error("The job is not empty, but hasn't been scrapped or waited for.");
    }
  }
}

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
      throw std::logic_error("The thread pool was stopped");
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

void Job::Scrap() {
  if (started_)
    throw std::logic_error("Cannot scrap a job that has already been started");
  tasks_.clear();
}

///////////////////////////////////////////////////////////////////////////

thread_local ThreadPoolBase *ThreadPoolBase::this_thread_pool_ = nullptr;
thread_local int ThreadPoolBase::this_thread_idx_ = -1;;

void ThreadPoolBase::Init(int num_threads) {
  if (shutdown_pending_)
    throw std::logic_error("The thread pool is being shut down.");
  std::lock_guard<std::mutex> g(mtx_);
  if (!threads_.empty())
    throw std::logic_error("The thread pool is already started!");
  threads_.reserve(num_threads);
  for (int i = 0; i < num_threads; i++)
    threads_.push_back(std::thread(&ThreadPoolBase::Run, this, i));
}

void ThreadPoolBase::Shutdown() {
  if (shutdown_pending_)
    return;
  {
    std::lock_guard<std::mutex> g(mtx_);
    if (shutdown_pending_)
      return;
    shutdown_pending_ = true;
    cv_.notify_all();
  }

  for (auto &t : threads_)
    t.join();

  assert(tasks_.empty());
}

void ThreadPoolBase::AddTask(TaskFunc f) {
  {
    std::lock_guard<std::mutex> g(mtx_);
    if (shutdown_pending_)
      throw std::logic_error("The thread pool is stopped and no longer accepts new tasks.");
    tasks_.push(std::move(f));
  }
  cv_.notify_one();
}

void ThreadPoolBase::Run(int index) noexcept {
  this_thread_pool_ = this;
  this_thread_idx_ = index;
  OnThreadStart(index);
  detail::CallAtExit([&]() { OnThreadStop(index); });
  std::unique_lock lock(mtx_);
  while (!shutdown_pending_ || !tasks_.empty()) {
    cv_.wait(lock, [&]() { return shutdown_pending_ || !tasks_.empty(); });
    if (tasks_.empty())
      break;
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
    if (tasks_.empty()) {
      assert(shutdown_pending_);
      return condition();
    }

    PopAndRunTask(lock);
  }
  return condition();
}


}  // namespace experimental
}  // namespace dali
