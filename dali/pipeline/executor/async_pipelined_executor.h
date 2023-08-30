// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_EXECUTOR_ASYNC_PIPELINED_EXECUTOR_H_
#define DALI_PIPELINE_EXECUTOR_ASYNC_PIPELINED_EXECUTOR_H_

#include <string>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/executor/pipelined_executor.h"
#include "dali/pipeline/util/worker_thread.h"

namespace dali {

/**
 * @brief This executor uses worker threads to pipelined the
 * issue of cpu, mixed, and gpu work. Calls the RunCPU,
 * RunMixed, and RunGPU are all asynchronous, and results
 * are retrieved by calling Outputs, which manages all
 * needed synchronization.
 */
class DLL_PUBLIC AsyncPipelinedExecutor : public PipelinedExecutor {
 public:
  DLL_PUBLIC inline AsyncPipelinedExecutor(int batch_size, int num_thread, int device_id,
                                           size_t bytes_per_sample_hint, bool set_affinity = false,
                                           int max_num_stream = -1,
                                           int default_cuda_stream_priority = 0,
                                           QueueSizes prefetch_queue_depth = QueueSizes{2, 2})
      : PipelinedExecutor(batch_size, num_thread, device_id, bytes_per_sample_hint, set_affinity,
                          max_num_stream, default_cuda_stream_priority, prefetch_queue_depth),
        cpu_thread_(device_id, set_affinity, "CPU executor"),
        mixed_thread_(device_id, set_affinity, "Mixed executor"),
        gpu_thread_(device_id, set_affinity, "GPU executor") {}

  DLL_PUBLIC ~AsyncPipelinedExecutor() override {
    Shutdown();
  }

  DLL_PUBLIC void Shutdown() override {
    ShutdownQueue();
    cpu_thread_.ForceStop();
    mixed_thread_.ForceStop();
    gpu_thread_.ForceStop();
    PipelinedExecutor::Shutdown();

    /*
     * We need to notify all work that may is scheduled that it should stop now. It may
     * happen that mixed and GPU stages are scheduled, but only GPU one is picked up while
     * mixed is discarded as the worker thread is already shutting down. In the end, GPU
     * state will wait infinitely for mixed one. This code defends against it.
     */
    mixed_work_cv_.notify_all();
    gpu_work_cv_.notify_all();
    /*
     * We need to call shutdown here and not rely on cpu_thread_ destructor
     * as when WorkerThread destructor is called conditional variables and mutexes
     * from this class may no longer exist while work inside WorkerThread is still
     * using it what can cause a hang
     */
    cpu_thread_.Shutdown();
    mixed_thread_.Shutdown();
    gpu_thread_.Shutdown();
  }

  DLL_PUBLIC void Init() override {
    if (!cpu_thread_.WaitForInit() || !mixed_thread_.WaitForInit() || !gpu_thread_.WaitForInit()) {
      cpu_thread_.ForceStop();
      mixed_thread_.ForceStop();
      gpu_thread_.ForceStop();
      std::string error = "Failed to init pipeline on device " + std::to_string(device_id_);
      throw std::runtime_error(error);
    }
  }

  DLL_PUBLIC void RunCPU() override;

  DLL_PUBLIC void RunMixed() override;

  DLL_PUBLIC void RunGPU() override;

  DLL_PUBLIC void Outputs(Workspace *ws) override {
    CheckForErrors();
    try {
      PipelinedExecutor::Outputs(ws);
    } catch (std::exception &e) {
      exec_error_ = true;
      SignalStop();
      mixed_work_cv_.notify_all();
      gpu_work_cv_.notify_all();
      throw;
    } catch (...) {
      exec_error_ = true;
      SignalStop();
      mixed_work_cv_.notify_all();
      gpu_work_cv_.notify_all();
      throw std::runtime_error("Unknown critical error in pipeline.");
    }
  }

 protected:
  void CheckForErrors() {
    cpu_thread_.CheckForErrors();
    mixed_thread_.CheckForErrors();
    gpu_thread_.CheckForErrors();
  }

  WorkerThread cpu_thread_, mixed_thread_, gpu_thread_;
  int cpu_work_counter_ = 0, mixed_work_counter_ = 0, gpu_work_counter_ = 0;
  std::condition_variable mixed_work_cv_, gpu_work_cv_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_ASYNC_PIPELINED_EXECUTOR_H_
