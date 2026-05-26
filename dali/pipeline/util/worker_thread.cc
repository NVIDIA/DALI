// Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/util/worker_thread.h"
#include "dali/pipeline/util/worker_thread_internal.h"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "dali/core/device_guard.h"
#include "dali/core/error_handling.h"
#include "dali/core/nvtx.h"

namespace dali {

void detail::Barrier::Wait(bool reset) {
  std::unique_lock<std::mutex> lock(mutex_);
  current_--;
  if (current_ == 0 || count_ == 0) {
    if (reset) {
      current_ = count_;
    }
    cv_.notify_all();
  } else {
    cv_.wait(lock, [this] { return current_ == 0; });
  }
}

void detail::Barrier::Break() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    count_ = 0;
    current_ = count_;
  }
  cv_.notify_all();
}

WorkerThreadImpl::WorkerThreadImpl(int device_id, bool set_affinity, std::string_view name) {
#if NVML_ENABLED
  if (set_affinity && device_id != CPU_ONLY_DEVICE_ID) {
    nvml_handle_ = nvml::NvmlInstance::CreateNvmlInstance();
  }
#endif
  thread_ = std::thread(&WorkerThreadImpl::ThreadMain,
                        this, device_id, set_affinity, make_string("[DALI][WT]", name));
}

WorkerThreadImpl::~WorkerThreadImpl() {
  Shutdown();
}

void WorkerThreadImpl::Shutdown() {
  // Wait for work to find errors
  if (running_) {
    WaitForWork(false);

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

void WorkerThreadImpl::DoWork(Work work) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    work_queue_.push(std::move(work));
    work_complete_ = false;
  }
  cv_.notify_one();
}

void WorkerThreadImpl::WaitForWork(bool rethrow_worker_errors) {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!work_complete_ && running_) {
    completed_.wait(lock);
  }

  // Check for errors
  if (rethrow_worker_errors && !errors_.empty()) {
    string error = "Error in worker thread: " + errors_.front();
    errors_.pop();
    running_ = false;
    lock.unlock();
    cv_.notify_all();
    throw std::runtime_error(error);
  }
}

void WorkerThreadImpl::ForceStop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
  }
  barrier_.Break();
  cv_.notify_all();
}

void WorkerThreadImpl::CheckForErrors() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!errors_.empty()) {
    string error = "Error in worker thread: " + errors_.front();
    errors_.pop();
    running_ = false;
    lock.unlock();
    cv_.notify_all();
    throw std::runtime_error(error);
  }
}

bool WorkerThreadImpl::WaitForInit() {
  barrier_.Wait();
  return running_;
}

void WorkerThreadImpl::ThreadMain(int device_id, bool set_affinity, const std::string &name) {
  SetThreadName(name.c_str());
  DeviceGuard g(device_id);
  try {
    if (set_affinity) {
#if NVML_ENABLED
      nvml::SetCPUAffinity();
#endif
    }
  } catch (std::exception &e) {
    std::lock_guard<std::mutex> lock(mutex_);
    errors_.push(e.what());
    running_ = false;
  } catch (...) {
    std::lock_guard<std::mutex> lock(mutex_);
    errors_.push("Unknown exception");
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
      std::cerr << std::this_thread::get_id() << " Exception in thread: " << e.what() << endl;
      lock.lock();
      errors_.push(e.what());
      running_ = false;
      completed_.notify_one();
      lock.unlock();
      break;
    } catch (...) {
      std::cerr << std::this_thread::get_id() << " Exception in thread" << endl;
      lock.lock();
      errors_.push("Caught unknown exception in thread.");
      running_ = false;
      completed_.notify_one();
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

std::unique_ptr<WorkerThread> CreateWorkerThread(
    int device_id, bool set_affinity, std::string_view name) {
  return std::make_unique<WorkerThreadImpl>(device_id, set_affinity, name);
}

}  // namespace dali
