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

// TODO(tgale):
// Document stream/event innefficiencies
// Document internal api improvements we could make
// - *OpNode could has accessors for important data in spec

namespace ndll {

// A helper class to manage a set of TensorLists and cudaEvents
template <typename Backend>
class TensorListPool {
public:
  inline TensorListPool(int size, EventPool *event_pool) {
    NDLL_ENFORCE(event_pool != nullptr);
    for (int i = 0; i < size; ++i) {
      tls_.push_back(std::make_shared<TensorList<Backend>>());
      tl_events_.push_back(event_pool->GetEvent());
    }
  }

  inline ~TensorListPool() = default;

  inline shared_ptr<TensorList<Backend>> GetTL(int idx) {
    return tls_[idx];
  }

  inline cudaEvent_t GetEvent(int idx) {
    return tl_events_[idx];
  }

  inline int size() const { return tls_.size(); }
  
private:
  vector<shared_ptr<TensorList<Backend>>> tls_;
  vector<cudaEvent_t> tl_events_;
};

class Executor {
public:
  inline Executor(int batch_size, int num_thread, int device_id,
      size_t bytes_per_sample_hint, bool set_affinity = false,
      int queue_depth = 2, int max_num_stream = -1) :
    batch_size_(batch_size), device_id_(device_id),
    bytes_per_sample_hint_(bytes_per_sample_hint),
    queue_depth_(queue_depth), stream_pool_(max_num_stream, true),
    event_pool_(max_num_stream), thread_pool_(num_thread, device_id, set_affinity) {
    NDLL_ENFORCE(batch_size_ > 0, "Batch size must be greater than 0.");
    NDLL_ENFORCE(device_id >= 0, "Device id must be non-negative.");
    NDLL_ENFORCE(queue_depth_ > 0, "Queue depth must be greater than 0.");
  }
  
  virtual ~Executor() = default;

  virtual void Build(OpGraph *graph, vector<string> output_names);

  virtual void RunCPU();

  virtual void RunInternal();

  virtual void RunGPU();

  virtual void Outputs(DeviceWorkspace *ws);
  
  friend class ExecutorTest;
  
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

  void SetupStreamsForGraph(OpGraph *graph,
      vector<internal::MixedWorkspace> *internal_data,
      vector<DeviceWorkspace> *gpu_data,
      StreamPool *stream_pool, EventPool *event_pool);

  void SetupOutputQueuesForGraph(
      const vector<string> &output_names, int queue_depth,
      EventPool *event_pool, OpGraph *graph, 
      std::map<string, int> *type_idx_map,
      vector<TensorListPool<CPUBackend>> *cpu_outputs,
      vector<TensorListPool<GPUBackend>> *gpu_outputs);

  void SetOutputBuffersForIter(
      const vector<string> &output_names,
      const std::map<string, int> &type_idx_map,
      int queue_idx, OpGraph *graph,
      vector<internal::MixedWorkspace> *internal_data,
      vector<DeviceWorkspace> *gpu_data,
      vector<TensorListPool<CPUBackend>> *cpu_outputs,
      vector<TensorListPool<GPUBackend>> *gpu_outputs);
  
  vector<HostWorkspace> cpu_op_data_;
  vector<internal::MixedWorkspace> internal_op_data_;
  vector<DeviceWorkspace> gpu_op_data_;

  // Used to keep track of the additional event insertions
  // we need to perform so that the user can block on the
  // results produced from a specific iteration
  using SyncPair = std::pair<cudaStream_t, cudaEvent_t>;
  vector<SyncPair> internal_output_events_, gpu_output_events_;
  
  int batch_size_, device_id_;
  size_t bytes_per_sample_hint_;

  vector<string> output_names_;
  std::map<string, int> type_idx_map_;
  vector<TensorListPool<CPUBackend>> cpu_outputs_;
  vector<TensorListPool<GPUBackend>> gpu_outputs_;
  int queue_depth_, queue_idx_ = 0;

  // The ready queue stores the indices of batches
  // who are ready for the user. We use the mutex
  // to ensure thread-safety while updating it,
  // and the condition_variable to signal between
  // the processing thread and the waiting host
  // thread that data is complete.
  std::queue<int> ready_queue_;  
  std::mutex ready_mutex_;
  std::condition_variable ready_cond_;
  
  OpGraph *graph_ = nullptr;
  StreamPool stream_pool_;
  EventPool event_pool_;
  ThreadPool thread_pool_;
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
