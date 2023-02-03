// Copyright (c) 2018-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>
#include <cstdlib>
#include <limits>
#include <utility>
#include "dali/pipeline/util/thread_pool.h"
#if NVML_ENABLED
#include "dali/util/nvml.h"
#endif
#include "dali/core/format.h"
#include "dali/core/cuda_error.h"
#include "dali/core/device_guard.h"
#include "dali/core/nvtx.h"

namespace dali {

ThreadPool::ThreadPool(int num_thread, int device_id, bool set_affinity, const char* name)
    : threads_(num_thread) {
  DALI_ENFORCE(num_thread > 0, "Thread pool must have non-zero size");
#if NVML_ENABLED
  // We use NVML only for setting thread affinity
  if (device_id != CPU_ONLY_DEVICE_ID && set_affinity) {
    nvml_handle_ = nvml::NvmlInstance::CreateNvmlInstance();
  }
#endif
  // Start the threads in the main loop
  for (int i = 0; i < num_thread; ++i) {
    threads_[i] = std::thread(std::bind(&ThreadPool::ThreadMain, this, i, device_id, set_affinity,
                                        make_string("[DALI][TP", i, "]", name)));
  }
  tl_errors_.resize(num_thread);
}

ThreadPool::~ThreadPool() {
  WaitForWork(false);

  std::unique_lock lock(queue_lock_);
  running_ = false;
  lock.unlock();
  // Each thread will lower the semaphore by at most 1
  queue_semaphore_.release(threads_.size());

  for (auto &thread : threads_) {
    thread.join();
  }
}

void ThreadPool::AddWork(Work work, int64_t priority, bool start_immediately) {
  bool started_before = started_;
  outstanding_work_.fetch_add(1);
  if (started_before) {
    std::lock_guard lock(queue_lock_);
    work_queue_.push({priority, std::move(work)});
  } else {
    work_queue_.push({priority, std::move(work)});
    if (start_immediately) {
      std::lock_guard lock(queue_lock_);
      started_ = true;
    }
  }
  if (started_) {
    if (started_before)
      queue_semaphore_.release();
    else
      queue_semaphore_.release(work_queue_.size());
  }
}

// Blocks until all work issued to the thread pool is complete
void ThreadPool::WaitForWork(bool checkForErrors) {
  if (outstanding_work_.load()) {
    std::unique_lock lock(completed_mutex_);
    completed_.wait(lock, [&, this] {
      return this->outstanding_work_ == 0;
    });
  }
  started_ = false;
  if (checkForErrors) {
    // Check for errors
    std::exception_ptr err;
    for (size_t i = 0; i < threads_.size(); ++i) {
      if (!err && !tl_errors_[i].empty()) {
        // Throw the first error that occurred
        err = std::move(tl_errors_[i].front());
      }
      tl_errors_[i] = {};
    }
    if (err)
      std::rethrow_exception(err);
  }
}

void ThreadPool::RunAll(bool wait) {
  if (!started_) {
    {
      std::lock_guard lock(queue_lock_);
      started_ = true;
    }
    queue_semaphore_.release(work_queue_.size());
  }
  if (wait) {
    WaitForWork();
  }
}

int ThreadPool::NumThreads() const {
  return threads_.size();
}

std::vector<std::thread::id> ThreadPool::GetThreadIds() const {
  std::vector<std::thread::id> tids;
  tids.reserve(threads_.size());
  for (const auto &thread : threads_)
    tids.emplace_back(thread.get_id());
  return tids;
}


void ThreadPool::ThreadMain(int thread_id, int device_id, bool set_affinity,
                            const std::string &name) {
  SetThreadName(name.c_str());
  DeviceGuard g(device_id);
  try {
#if NVML_ENABLED
    if (set_affinity) {
      const char *env_affinity = std::getenv("DALI_AFFINITY_MASK");
      int core = -1;
      if (env_affinity) {
        const auto &vec = string_split(env_affinity, ',');
        if ((size_t)thread_id < vec.size()) {
          core = std::stoi(vec[thread_id]);
        } else {
          DALI_WARN("DALI_AFFINITY_MASK environment variable is set, "
                    "but does not have enough entries: thread_id (", thread_id,
                    ") vs #entries (", vec.size(), "). Ignoring...");
        }
      }
      nvml::SetCPUAffinity(core);
    }
#endif
  } catch (...) {
    tl_errors_[thread_id].push(std::current_exception());
  }

  while (running_) {
    // Wait for something to do
    queue_semaphore_.acquire();

    // This lock guards only the queue, not the condition - that's handled by the semaphore
    std::unique_lock lock(queue_lock_);

    if (!running_)
      break;

    // Get work from the queue.
    Work work = std::move(work_queue_.top().second);
    work_queue_.pop();
    // Unlock the lock
    lock.unlock();

    // If an error occurs, we save it in tl_errors_. When
    // WaitForWork is called, we will check for any errors
    // in the threads and return an error if one occured.
    try {
      work(thread_id);
    } catch (...) {
      tl_errors_[thread_id].push(std::current_exception());
    }

    // The task is now complete - we can atomically decrement the number of outstanding work.
    // If it reaches zero, we must safely notify the potential threads waiting for the work
    // to complete.
    // NOTE: We don't have to acquire the mutex until the number of waiting threads reaches 0.
    if (--outstanding_work_ == 0) {
      // We don't need to guard the modification of the atomic value with a mutex -
      // however, we need to lock it briefly to make sure we don't have this scenario:
      //
      // worker                           WaitForWork
      //
      //                                  completed_mutex_.lock()
      //                                  return outstanding_work_ == 0  (false!)
      // --outstanding_work == 0 (true)
      // compleded_.notify_all()          NOT WAITING FOR compleded_ YET!!!!!!!!!!!!!
      //                                  atomically unlock `lock` and wait for `completed_`
      //                                                               ^^^^ deadlock


      // The brief lock/unlock sequence avoids the above.
      // The call to lock.lock() prevents the worker thread from signalling the event while
      // the control thread is evaluating the condition (which happens with the mutex owned).
      // Now it looks like this:
      //
      // worker                           WaitForWork
      //
      //                                  completed_mutex_.lock()
      //                                  return outstanding_work_ == 0  (false!)
      // --outstanding_work == 0 (true)
      // completed_mutex_.lock()
      //                                  atomically unlock `lock` and wait for `completed_`
      // At this point we know that if
      // anyone was executing WaitForWork
      // they're not evaluating the
      // condition but rather waiting on
      // the completed_ condvar.
      //
      // completed_mutex_.unlock()
      // compleded_.notify_all()
      //                                  notified - wake up
      //                                  completed_mutex_.lock()
      //                                  continue execution
      {
        std::lock_guard lock2(completed_mutex_);
      }
      completed_.notify_all();
    }
  }
}

}  // namespace dali
