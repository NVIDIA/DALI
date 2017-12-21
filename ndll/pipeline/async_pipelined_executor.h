// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_ASYNC_PIPELINED_EXECUTOR_H_
#define NDLL_PIPELINE_ASYNC_PIPELINED_EXECUTOR_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/pipelined_executor.h"
#include "ndll/pipeline/util/worker_thread.h"

namespace ndll {

/**
 * @brief This executor uses worker threads to pipelined the
 * issue of cpu, internal, and gpu work. Calls the RunCPU,
 * RunInternal, and RunGPU are all asynchronous, and results
 * are retrieved by calling Outputs, which manages all
 * needed synchronization.
 */
class AsyncPipelinedExecutor : public PipelinedExecutor {
 public:
  inline AsyncPipelinedExecutor(int batch_size, int num_thread,
      int device_id, size_t bytes_per_sample_hint,
      bool set_affinity = false, int max_num_stream = -1) :
    PipelinedExecutor(batch_size, num_thread, device_id,
        bytes_per_sample_hint, set_affinity, max_num_stream),
    cpu_thread_(device_id, set_affinity),
    internal_thread_(device_id, set_affinity),
    gpu_thread_(device_id, set_affinity) {}

  virtual ~AsyncPipelinedExecutor() = default;

  void RunCPU() override;

  void RunInternal() override;

  void RunGPU() override;

  void Outputs(DeviceWorkspace *ws) override {
    CheckForErrors();
    PipelinedExecutor::Outputs(ws);
  }

 protected:
  void CheckForErrors() {
    cpu_thread_.CheckForErrors();
    internal_thread_.CheckForErrors();
    gpu_thread_.CheckForErrors();
  }

  WorkerThread cpu_thread_, internal_thread_, gpu_thread_;
  int cpu_work_counter_ = 0, internal_work_counter_ = 0, gpu_work_counter_ = 0;
  std::mutex cpu_mutex_, internal_mutex_, gpu_mutex_;
  std::condition_variable internal_work_cv_, gpu_work_cv_;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_ASYNC_PIPELINED_EXECUTOR_H_
