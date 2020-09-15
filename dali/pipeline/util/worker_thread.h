// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_UTIL_WORKER_THREAD_H_
#define DALI_PIPELINE_UTIL_WORKER_THREAD_H_

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#if NVML_ENABLED
#include "dali/util/nvml.h"
#endif
#include "dali/core/device_guard.h"

namespace dali {

class Barrier {
 public:
    explicit Barrier(std::size_t count) : count_(count), current_(count) {}
    void Wait(bool reset = false) {
        std::unique_lock<std::mutex> lock(mutex_);
        current_--;
        if (current_ == 0 || count_ == 0) {
            if (reset)
              current_ = count_;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [this] { return current_ == 0; });
        }
    }
    void Break() {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        count_ = 0;
        current_ = count_;
      }
      cv_.notify_all();
    }

 private:
    std::mutex mutex_;
    std::condition_variable cv_;
    size_t count_;
    size_t current_;
};

class WorkerThread {
 public:
  typedef std::function<void(void)> Work;

  inline WorkerThread(int device_id, bool set_affinity) :
    running_(true), work_complete_(true), barrier_(2) {
#if NVML_ENABLED
    if (device_id != CPU_ONLY_DEVICE_ID) {
      nvml::Init();
    }
#endif
    thread_ = std::thread(&WorkerThread::ThreadMain,
        this, device_id, set_affinity);
  }

  inline ~WorkerThread() {
#if NVML_ENABLED
    nvml::Shutdown();
#endif
  }

  /*
   * Separate Shutdown function is needed as we cannot rely on destructor to stop the thread.
   * When the destructor is called other things that work() is using may have been gone long
   * before causing a hang. Now when Shutdown is called we are sure that all things around still exist.
   */
  inline void Shutdown(void) {
    // Wait for work to find errors
    if (running_) {
      WaitForWork();

      // Mark the thread as not running
      {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
      }
      cv_.notify_one();
    } else {
      ForceStop();
    }
    // Join the thread
    if (thread_.joinable()) {
      ForceStop();
      thread_.join();
    }
  }

  inline void DoWork(Work work) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      work_queue_.push(std::move(work));
      work_complete_ = false;
    }
    cv_.notify_one();
  }

  inline void WaitForWork() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!work_complete_) {
      completed_.wait(lock);
    }

    // Check for errors
    if (!errors_.empty()) {
      string error = "Error in worker thread: " +
        errors_.front();
      errors_.pop();
      running_ = false;
      lock.unlock();
      cv_.notify_all();
      throw std::runtime_error(error);
    }
  }

  inline void ForceStop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      running_ = false;
    }
    barrier_.Break();
    cv_.notify_all();
  }

  inline void CheckForErrors() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!errors_.empty()) {
      string error = "Error in worker thread: " +
        errors_.front();
      errors_.pop();
    }
  }

  inline bool WaitForInit() {
    barrier_.Wait();
    return running_;
  }

 private:
  void ThreadMain(int device_id, bool set_affinity) {
    DeviceGuard g(device_id);
    try {
      if (set_affinity) {
#if NVML_ENABLED
        nvml::SetCPUAffinity();
#endif
      }
    } catch (std::exception &e) {
      errors_.push(e.what());
      std::lock_guard<std::mutex> lock(mutex_);
      running_ = false;
    } catch (...) {
      errors_.push("Unknown exception");
      std::lock_guard<std::mutex> lock(mutex_);
      running_ = false;
    }

    barrier_.Wait();

    while (running_) {
      // Check the queue for work
      std::unique_lock<std::mutex> lock(mutex_);
      while (work_queue_.empty() && running_) {
        cv_.wait(lock);
      }

      if (!running_) {
        break;
      }

      Work work = std::move(work_queue_.front());
      work_queue_.pop();
      lock.unlock();

      try {
        work();
      } catch (std::exception &e) {
        cout << std::this_thread::get_id() << " Exception in thread: " << e.what() << endl;
        lock.lock();
        errors_.push(e.what());
        running_ = false;
        lock.unlock();
        break;
      } catch (...) {
        cout << std::this_thread::get_id() << " Exception in thread" << endl;
        lock.lock();
        errors_.push("Caught unknown exception in thread.");
        running_ = false;
        lock.unlock();
        break;
      }

      lock.lock();

      if (work_queue_.empty()) {
        work_complete_ = true;
        completed_.notify_one();
      }
    }
  }

  bool running_, work_complete_;
  std::queue<Work> work_queue_;
  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable cv_, completed_;

  std::queue<string> errors_;

  Barrier barrier_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_WORKER_THREAD_H_
