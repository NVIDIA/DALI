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

#ifndef DALI_PIPELINE_EXECUTOR_EXECUTOR_H_
#define DALI_PIPELINE_EXECUTOR_EXECUTOR_H_

#include <utility>
#include <vector>
#include <string>
#include <queue>
#include <map>
#include <memory>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/workspace/device_workspace.h"
#include "dali/pipeline/workspace/host_workspace.h"
#include "dali/pipeline/workspace/mixed_workspace.h"
#include "dali/pipeline/workspace/support_workspace.h"
#include "dali/pipeline/op_graph.h"
#include "dali/pipeline/util/event_pool.h"
#include "dali/pipeline/util/stream_pool.h"
#include "dali/pipeline/util/thread_pool.h"

namespace dali {

/**
 * @brief Basic executor for dali graphs. This executor enables
 * prefetching of results by maintaining two copies of output
 * buffers, so that we can produce data into one while the 
 * other is in use by the user.
 */
class DLL_PUBLIC Executor {
 public:
  DLL_PUBLIC inline Executor(int batch_size, int num_thread, int device_id,
      size_t bytes_per_sample_hint, bool set_affinity = false,
      int max_num_stream = -1) :
    batch_size_(batch_size), device_id_(device_id),
    bytes_per_sample_hint_(bytes_per_sample_hint),
    queue_depth_(2),
    stream_pool_(max_num_stream, true), event_pool_(max_num_stream),
    thread_pool_(num_thread, device_id, set_affinity),
    exec_error_(false) {
    DALI_ENFORCE(batch_size_ > 0, "Batch size must be greater than 0.");
    DALI_ENFORCE(device_id >= 0, "Device id must be non-negative.");
  }

  DLL_PUBLIC virtual ~Executor() = default;

  DLL_PUBLIC virtual void Build(OpGraph *graph, vector<string> output_names);

  DLL_PUBLIC virtual void Init() {}

  DLL_PUBLIC virtual void RunCPU();

  DLL_PUBLIC virtual void RunMixed();

  DLL_PUBLIC virtual void RunGPU();

  DLL_PUBLIC virtual void Outputs(DeviceWorkspace *ws);

  friend class ExecutorTest;

  DISABLE_COPY_MOVE_ASSIGN(Executor);

 protected:
  using WorkspaceBlob = struct {
    vector<HostWorkspace> cpu_op_data;
    vector<MixedWorkspace> mixed_op_data;
    vector<DeviceWorkspace> gpu_op_data;
    vector<SupportWorkspace> support_op_data;

    void Clear() {
      cpu_op_data.clear();
      mixed_op_data.clear();
      gpu_op_data.clear();
      support_op_data.clear();
    }
  };
  vector<WorkspaceBlob> wss_;

  void PruneUnusedGraphNodes();

  void SetupDataForGraph(WorkspaceBlob *wsb);

  void PresizeData(WorkspaceBlob *wsb);

  void SetupStreamsForGraph(WorkspaceBlob *wsb);

  void SetupOutputQueuesForGraph();

  void SetOutputBuffersForIter(int queue_idx, WorkspaceBlob *wsb);

  template <typename Backend>
  class TensorListPool {
   public:
    inline TensorListPool(int size, int batch_size, size_t bytes_hint) {
      for (int i = 0; i < size; ++i) {
        tls_.push_back(std::make_shared<TensorList<Backend>>());
        tls_.back()->Resize({{batch_size*(Index)bytes_hint}});
      }
    }

    inline shared_ptr<TensorList<Backend>> Get(int idx) {
      return tls_[idx];
    }
   private:
    vector<shared_ptr<TensorList<Backend>>> tls_;
  };

  class EventList {
   public:
    inline EventList() {}
    inline EventList(int size, EventPool *event_pool) {
      DALI_ENFORCE(event_pool != nullptr);
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

  int batch_size_, device_id_;
  size_t bytes_per_sample_hint_;
  int queue_depth_;
  int previous_gpu_queue_idx_ = -1;

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

  // Buffers are rotated between being 'free', where the
  // pipeline is ok to fill them with data, 'ready', where
  // they are already full of prepared data, and 'in-use',
  // where the user currently owns that buffer. A buffer
  // is marked as in-use when it is returned as and output.
  // The buffer is then returned the the ready queue the
  // next time Ouputs() is called.
  std::queue<int> ready_queue_, free_queue_, in_use_queue_;
  std::mutex ready_mutex_, free_mutex_;
  std::condition_variable ready_cond_, free_cond_;

  // Work is passed between the stages through queues. This
  // is needed for potentially asynchronous work issue, which
  // some Executors that derive from this class implement.
  //
  // In the case that work issue is pipelined, a stages issue
  // could run at the same time as the next iterations issue
  // for the previous stage. To avoid thread-safety issues
  // with updating our queues, we need to lock when we update
  // them. However, this executor assumes the same thread
  // will call Run*, so it does not block if no work exists
  // for the stage that was called (it will throw an error).
  //
  // Derived executors that implement asynchronous work issue
  // must handle their own synchronization between the same
  // iteration of each stage. While it is not ideal to have
  // two sets of locks doing similar things in each stage,
  // it simplifies the software for now so we leave it
  // unless it becomes an issue in the future.
  std::queue<int> mixed_work_queue_, gpu_work_queue_;
  std::mutex mixed_mutex_, gpu_mutex_;

  OpGraph *graph_ = nullptr;
  StreamPool stream_pool_;
  EventPool event_pool_;
  ThreadPool thread_pool_;
  std::vector<std::string> errors_;
  std::mutex errors_mutex_;
  bool exec_error_;
};

#define USE_EXECUTOR_MEMBERS()                             \
  protected:                                               \
  using Executor::WorkspaceBlob;                           \
  using Executor::wss_;                                    \
  using Executor::batch_size_;                             \
  using Executor::device_id_;                              \
  using Executor::bytes_per_sample_hint_;                  \
  using Executor::queue_depth_;                            \
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

}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR_H_
