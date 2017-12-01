#ifndef NDLL_PIPELINE_EXECUTOR_H_
#define NDLL_PIPELINE_EXECUTOR_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/device_workspace.h"
#include "ndll/pipeline/host_workspace.h"
#include "ndll/pipeline/mixed_workspace.h"
#include "ndll/pipeline/op_graph.h"
#include "ndll/pipeline/util/event_pool.h"
#include "ndll/pipeline/util/stream_pool.h"
#include "ndll/pipeline/util/thread_pool.h"

namespace ndll {

class Executor {
public:
  inline Executor(int batch_size, int device_id, size_t bytes_per_sample_hint,
      int max_num_stream = -1) :
    batch_size_(batch_size), device_id_(device_id),
    bytes_per_sample_hint_(bytes_per_sample_hint),
    stream_pool_(max_num_stream, true),
    event_pool_(max_num_stream) {
    NDLL_ENFORCE(batch_size_ > 0, "Batch size must be greater than 0.");
    NDLL_ENFORCE(device_id > 0, "Device id must be greater than 0.");
  }
  
  virtual ~Executor() = default;

  virtual void Build(OpGraph *graph,
      vector<string> output_names);

  virtual void RunCPU() = 0;

  virtual void RunInternal() = 0;

  virtual void RunGPU() = 0;
  
  DISABLE_COPY_MOVE_ASSIGN(Executor);
protected:
  // Return the nearest multiple of 8 that is >= base_ptr_offset
  inline size_t round_up_to_8(size_t base_ptr_offset) {
    if (base_ptr_offset & 7) {
      base_ptr_offset = (base_ptr_offset & ~7) + 8;
    }
    return base_ptr_offset;
  }

  void PruneUnusedGraphNodes(OpGraph *graph,
      vector<string> output_names);
  
  void SetupDataForGraph(OpGraph *graph,
      vector<HostWorkspace> *cpu_data,
      vector<internal::MixedWorkspace> *internal_data,
      vector<DeviceWorkspace> *gpu_data);

  void PresizeData(vector<HostWorkspace> *cpu_data,
      vector<internal::MixedWorkspace> *internal_data,
      vector<DeviceWorkspace> *gpu_data,
      size_t bytes_per_sample_hint);

  void SetupMegaBufferForGraph(OpGraph *graph,
      Tensor<CPUBackend> *mega_buffer,
      Tensor<GPUBackend> *mega_buffer_gpu,
      vector<DeviceWorkspace> *gpu_data);

  void SetupStreamsForGraph(OpGraph *graph,
      vector<cudaEvent_t> *gpu_op_events,
      vector<vector<cudaEvent_t>> *gpu_op_parent_events,
      vector<DeviceWorkspace> *gpu_data,
      StreamPool *stream_pool, EventPool *event_pool);
  
  vector<HostWorkspace> cpu_op_data_;
  vector<internal::MixedWorkspace> internal_op_data_;
  vector<DeviceWorkspace> gpu_op_data_;

  Tensor<CPUBackend> mega_buffer_;
  Tensor<GPUBackend> mega_buffer_gpu_;
  
  int batch_size_, device_id_;
  size_t bytes_per_sample_hint_;

  vector<cudaEvent_t> gpu_op_events_;
  vector<vector<cudaEvent_t>> gpu_op_parent_events_;

  StreamPool stream_pool_;
  EventPool event_pool_;
};

#define USE_EXECUTOR_MEMBERS()                             \
  using Executor::cpu_op_data_;                            \
  using Executor::internal_op_data_;                       \
  using Executor::gpu_op_data_;                            \
  using Executor::batch_size_;                             \
  using Executor::device_id_;                              \
  using Executor::bytes_per_sample_hint_
  
  
// class ThreadedExecutor : Executor {
// public:
//   inline ThreadedExecutor(int batch_size, int device_id, size_t bytes_per_sample_hint,
//       int num_threads, bool set_affinity) :
//     Executor(batch_size, device_id, bytes_per_sample_hint), 
//     thread_pool_(num_threads, device_id, set_affinity) {}
  
//   inline ThreadedExecutor(OpGraph *graph, int batch_size, int device_id,
//       size_t bytes_per_sample_hint, int num_threads, bool set_affinity) :
//     Executor(batch_size, device_id, bytes_per_sample_hint),
//     thread_pool_(num_threads, device_id, set_affinity) {
//     NDLL_ENFORCE(graph != nullptr, "Graph cannot be nullptr.");
//     Build(graph);
//   }
  
//   virtual ~ThreadedExecutor() = default;

//   void Build(OpGraph *graph) override;

//   void RunCPU() override;

//   void RunInternal() override;

//   void RunGPU() override;

//   DISABLE_COPY_MOVE_ASSIGN(ThreadedExecutor);
// protected:
//   ThreadPool thread_pool_;

//   USE_EXECUTOR_MEMBERS();
// };

} // namespace ndll

#endif // NDLL_PIPELINE_EXECUTOR_H_
