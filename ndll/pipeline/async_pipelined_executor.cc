// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/async_pipelined_executor.h"

namespace ndll {

void AsyncPipelinedExecutor::RunCPU() {
  std::unique_lock<std::mutex> lock(cpu_mutex_);
  ++cpu_work_counter_;
  lock.unlock();
  cpu_thread_.DoWork([this]() {
        // Run the cpu work. We know there is cpu
        // work so we do not have to wait to take
        // the work
        std::unique_lock<std::mutex> lock(cpu_mutex_);
        NDLL_ENFORCE(cpu_work_counter_ > 0,
            "Internal error, thread has no cpu work.");
        --cpu_work_counter_;
        lock.unlock();

        // cout << "got cpu work" << endl;
        PipelinedExecutor::RunCPU();
        // cout << "finished cpu work" << endl;

        // Mark that there is now internal work to do
        // and signal to any threads that are waiting
        std::unique_lock<std::mutex> internal_lock(internal_mutex_);
        ++internal_work_counter_;
        internal_work_cv_.notify_one();
      });
}

void AsyncPipelinedExecutor::RunInternal() {
  internal_thread_.DoWork([this]() {
        // Block until there is internal work to do
        std::unique_lock<std::mutex> lock(internal_mutex_);
        while (internal_work_counter_ == 0) {
          internal_work_cv_.wait(lock);
        }
        --internal_work_counter_;
        lock.unlock();

        // cout << "got internal work" << endl;
        PipelinedExecutor::RunInternal();
        // cout << "finished internal issue" << endl;

        // Mark that there is now gpu work to do
        // and signal to any threads that are waiting
        std::unique_lock<std::mutex> gpu_lock(gpu_mutex_);
        ++gpu_work_counter_;
        gpu_work_cv_.notify_one();
        gpu_lock.unlock();
      });
}

void AsyncPipelinedExecutor::RunGPU() {
  gpu_thread_.DoWork([this]() {
        // Block until there is gpu work to do
        std::unique_lock<std::mutex> lock(gpu_mutex_);
        while (gpu_work_counter_ == 0) {
          gpu_work_cv_.wait(lock);
        }
        --gpu_work_counter_;
        lock.unlock();

        // cout << "got gpu work" << endl;
        PipelinedExecutor::RunGPU();
        // cout << "Finished gpu issue" << endl;

        // All the work for this batch has now been issued,
        // but has not necessarilly finished. The base-class
        // handles any synchronization for output completion
      });
}

}  // namespace ndll
