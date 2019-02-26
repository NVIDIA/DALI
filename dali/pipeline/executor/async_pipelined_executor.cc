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

#include "dali/pipeline/executor/async_pipelined_executor.h"

namespace dali {

void AsyncPipelinedExecutor::RunCPU() {
  CheckForErrors();
  std::unique_lock<std::mutex> lock(cpu_mutex_);
  ++cpu_work_counter_;
  lock.unlock();
  cpu_thread_.DoWork([this]() {
        // Run the cpu work. We know there is cpu
        // work so we do not have to wait to take
        // the work
        std::unique_lock<std::mutex> lock(cpu_mutex_);
        DALI_ENFORCE(cpu_work_counter_ > 0,
            "Internal error, thread has no cpu work.");
        --cpu_work_counter_;
        lock.unlock();

        if (exec_error_) {
          mixed_work_cv_.notify_all();
          return;
        }
        PipelinedExecutor::RunCPU();

        // Mark that there is now mixed work to do
        // and signal to any threads that are waiting
        std::unique_lock<std::mutex> mixed_lock(mixed_mutex_);
        ++mixed_work_counter_;
        mixed_work_cv_.notify_one();
      });
}

void AsyncPipelinedExecutor::RunMixed() {
  CheckForErrors();
  mixed_thread_.DoWork([this]() {
        // Block until there is mixed work to do
        std::unique_lock<std::mutex> lock(mixed_mutex_);
        while (mixed_work_counter_ == 0 && !exec_error_) {
          mixed_work_cv_.wait(lock);
        }
        --mixed_work_counter_;
        lock.unlock();
        if (exec_error_) {
          gpu_work_cv_.notify_all();
          return;
        }

        PipelinedExecutor::RunMixed();

        // Mark that there is now gpu work to do
        // and signal to any threads that are waiting
        std::unique_lock<std::mutex> gpu_lock(gpu_mutex_);
        ++gpu_work_counter_;
        gpu_work_cv_.notify_one();
        gpu_lock.unlock();
      });
}

void AsyncPipelinedExecutor::RunGPU() {
  CheckForErrors();
  gpu_thread_.DoWork([this]() {
        // Block until there is gpu work to do
        std::unique_lock<std::mutex> lock(gpu_mutex_);
        while (gpu_work_counter_ == 0 && !exec_error_) {
          gpu_work_cv_.wait(lock);
        }
        --gpu_work_counter_;
        lock.unlock();
        if (exec_error_)
          return;

        PipelinedExecutor::RunGPU();

        // All the work for this batch has now been issued,
        // but has not necessarilly finished. The base-class
        // handles any synchronization for output completion
      });
}

}  // namespace dali
