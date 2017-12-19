#ifndef NDLL_PIPELINE_ASYNC_PIPELINED_EXECUTOR_H_
#define NDLL_PIPELINE_ASYNC_PIPELINED_EXECUTOR_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/pipelined_executor.h"

namespace ndll {

// Note: RunCPU, RunGPU, and RunInternal are all asynchronous.
// They just put work in a thread pool, and the work manages
// all synchronization between different stages of the pipeline.
// Could this deadlock at all? We should have checks to make sure
// the methods are called in order...
class AsyncPipelinedExecutor : public PipelinedExecutor {
public:
  inline AsyncPipelinedExecutor(int batch_size, int num_thread,
      int device_id, size_t bytes_per_sample_hint,
      bool set_affinity = false, int max_num_stream = -1) :
    PipelinedExecutor(batch_size, num_thread, device_id,
        bytes_per_sample_hint, set_affinity, max_num_stream),
    issue_threads_(3, device_id, set_affinity) {}

  virtual ~AsyncPipelinedExecutor() = default;

  void RunCPU() override;

  void RunInternal() override;

  void RunGPU() override;

protected:  
  ThreadPool issue_threads_;
  int cpu_work_counter_ = 0, internal_work_counter_ = 0, gpu_work_counter_ = 0;
  std::mutex cpu_mutex_, internal_mutex_, gpu_mutex_;
  std::condition_variable internal_work_cv_, gpu_work_cv_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_ASYNC_PIPELINED_EXECUTOR_H_
