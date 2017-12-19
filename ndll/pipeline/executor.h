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

class Executor {
public:
  inline Executor(int batch_size, int num_thread, int device_id,
      size_t bytes_per_sample_hint, bool set_affinity = false,
      int max_num_stream = -1) :
    batch_size_(batch_size), device_id_(device_id),
    bytes_per_sample_hint_(bytes_per_sample_hint),
    queue_depth_(2), stream_pool_(max_num_stream, true),
    event_pool_(max_num_stream), thread_pool_(num_thread, device_id, set_affinity) {
    NDLL_ENFORCE(batch_size_ > 0, "Batch size must be greater than 0.");
    NDLL_ENFORCE(device_id >= 0, "Device id must be non-negative.");
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
  
  void PruneUnusedGraphNodes();
  
  void SetupDataForGraph();

  void PresizeData();

  void SetupStreamsForGraph();

  void SetupOutputQueuesForGraph();
  
  void SetOutputBuffersForIter();

  template <typename Backend>
  class TensorListPool {
  public:
    inline TensorListPool(int size, int batch_size, size_t bytes_hint) {
      for (int i = 0; i < size; ++i) {
        tls_.push_back(std::make_shared<TensorList<Backend>>());
        tls_.back()->Resize({{batch_size*(Index)bytes_hint}});
      }
    }

    inline shared_ptr<TensorList<Backend>> GetTL(int idx) {
      return tls_[idx];
    }
  private:
    vector<shared_ptr<TensorList<Backend>>> tls_;
  };

  class EventList {
  public:
    inline EventList() {}
    inline EventList(int size, EventPool *event_pool) {
      NDLL_ENFORCE(event_pool != nullptr);
      for (int i = 0; i < size; ++i) {
        events_.push_back(event_pool->GetEvent());
      }      
    }
    
    inline cudaEvent_t GetEvent(int idx) {
      return events_[idx];
    }
  private:
    vector<cudaEvent_t> events_;
  };

  vector<HostWorkspace> cpu_op_data_;
  vector<internal::MixedWorkspace> internal_op_data_;
  vector<DeviceWorkspace> gpu_op_data_;

  int batch_size_, device_id_;
  size_t bytes_per_sample_hint_;
  int queue_depth_, queue_idx_ = 0;
  
  vector<string> output_names_;
  std::map<string, int> type_idx_map_;
  vector<TensorListPool<CPUBackend>> cpu_outputs_;
  vector<TensorListPool<GPUBackend>> gpu_outputs_;
  vector<EventList> gpu_output_events_;

  // Meta-data about our stage outputs for fast lookup
  using OutputInfo = struct {
    std::pair<NodeID, int> prod_and_idx;
    vector<std::pair<NodeID, int>> con_and_idx;
  };
  vector<OutputInfo> cpu_output_info_, gpu_output_info_;
  
  
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
  protected:                                               \
  using Executor::cpu_op_data_;                            \
  using Executor::internal_op_data_;                       \
  using Executor::gpu_op_data_;                            \
  using Executor::batch_size_;                             \
  using Executor::device_id_;                              \
  using Executor::bytes_per_sample_hint_;                  \
  using Executor::queue_depth_;                            \
  using Executor::queue_idx_;                              \
  using Executor::output_names_;                           \
  using Executor::type_idx_map_;                           \
  using Executor::cpu_outputs_;                            \
  using Executor::gpu_outputs_;                            \
  using Executor::gpu_output_events_;                      \
  using Executor::OutputInfo;                              \
  using Executor::cpu_output_info_;                        \
  using Executor::gpu_output_info_;                        \
  using Executor::ready_queue_;                            \
  using Executor::ready_mutex_;                            \
  using Executor::ready_cond_;                             \
  using Executor::graph_;                                  \
  using Executor::stream_pool_;                            \
  using Executor::event_pool_;                             \
  using Executor::thread_pool_
  

} // namespace ndll

#endif // NDLL_PIPELINE_EXECUTOR_H_
