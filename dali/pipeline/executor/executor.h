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

#include <map>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/graph/op_graph.h"
#include "dali/pipeline/util/event_pool.h"
#include "dali/pipeline/util/stream_pool.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/pipeline/workspace/device_workspace.h"
#include "dali/pipeline/workspace/host_workspace.h"
#include "dali/pipeline/workspace/mixed_workspace.h"
#include "dali/pipeline/workspace/support_workspace.h"

#include "dali/pipeline/executor/workspace_policy.h"
#include "dali/pipeline/graph/op_graph_verifier.h"
#include "dali/pipeline/workspace/workspace_data_factory.h"

namespace dali {

static DALIOpType PreviousStage(DALIOpType op) {
  switch (op) {
    case DALIOpType::CPU:
      return DALIOpType::SUPPORT;
    case DALIOpType::MIXED:
      return DALIOpType::CPU;
    case DALIOpType::GPU:
      return DALIOpType::MIXED;
    default:
      return static_cast<DALIOpType>(-1);  // No previous OpType
  }
}

static bool HasPreviousStage(DALIOpType op) {
  if (op == DALIOpType::SUPPORT) {
    return false;
  }
  return true;
}

static DALIOpType NextStage(DALIOpType op) {
  switch (op) {
    case DALIOpType::SUPPORT:
      return DALIOpType::CPU;
    case DALIOpType::CPU:
      return DALIOpType::MIXED;
    case DALIOpType::MIXED:
      return DALIOpType::GPU;
    default:
      return static_cast<DALIOpType>(-1);  // No next OpType
  }
}

static bool HasNextStage(DALIOpType op) {
  if (op == DALIOpType::GPU) {
    return false;
  }
  return true;
}


struct OutputIdxs {
  int mixed;
  int gpu;

  int &operator[](DALIOpType op_type) {
    if (op_type == DALIOpType::MIXED) {
      return mixed;
    }
    else {
      return gpu;
    }
  }

  const int &operator[](DALIOpType op_type) const {
    if (op_type == DALIOpType::MIXED) {
      return mixed;
    }
    else {
      return gpu;
    }
  }
};

/**
 * @brief Basic executor for dali graphs. This executor enables
 * prefetching of results by maintaining two copies of output
 * buffers, so that we can produce data into one while the
 * other is in use by the user.
 */
class DLL_PUBLIC Executor : public JIT_WS_Policy {
 public:
  using ExecutorCallback = std::function<void(void)>;

  DLL_PUBLIC inline Executor(int batch_size, int num_thread, int device_id,
                             size_t bytes_per_sample_hint, bool set_affinity = false,
                             int max_num_stream = -1, QueueSizes prefetch_queue_depth = 2)
      : batch_size_(batch_size),
        device_id_(device_id),
        bytes_per_sample_hint_(bytes_per_sample_hint),
        // queue_depth_(prefetch_queue_depth),
        stream_pool_(max_num_stream, true),
        event_pool_(max_num_stream),
        thread_pool_(num_thread, device_id, set_affinity),
        exec_error_(false),
        cb_(nullptr),
        queue_sizes_(prefetch_queue_depth) {
    DALI_ENFORCE(batch_size_ > 0, "Batch size must be greater than 0.");
    DALI_ENFORCE(device_id >= 0, "Device id must be non-negative.");

    stage_queue_depths_[static_cast<int>(DALIOpType::SUPPORT)] = 1; // synchronous with CPU
    stage_queue_depths_[static_cast<int>(DALIOpType::CPU)] = prefetch_queue_depth.cpu_size;
    stage_queue_depths_[static_cast<int>(DALIOpType::MIXED)] = prefetch_queue_depth.mixed_size;
    stage_queue_depths_[static_cast<int>(DALIOpType::GPU)] = prefetch_queue_depth.gpu_size;
  }

  DLL_PUBLIC virtual ~Executor() = default;

  DLL_PUBLIC virtual void Build(OpGraph *graph, vector<string> output_names);

  DLL_PUBLIC virtual void Init() {}

  DLL_PUBLIC virtual void RunCPU();

  DLL_PUBLIC virtual void RunMixed();

  DLL_PUBLIC virtual void RunGPU();

  DLL_PUBLIC virtual void Outputs(DeviceWorkspace *ws);

  DLL_PUBLIC virtual void ShareOutputs(DeviceWorkspace *ws);

  DLL_PUBLIC virtual void ReleaseOutputs();

  DLL_PUBLIC virtual void SetCompletionCallback(ExecutorCallback cb);

  friend class ExecutorTest;

  DISABLE_COPY_MOVE_ASSIGN(Executor);

 protected:
  using JIT_WS_Policy::InitializeWorkspaceStore;
  using JIT_WS_Policy::GetWorkspace;

 



  void PruneUnusedGraphNodes();

  virtual std::vector<int> GetTensorQueueSizes(const OpGraph &graph);

  void SetupWorkspacesForGraph(int queue_idx);

  virtual void SetupOutputInfo(const OpGraph &graph);

  std::vector<int> GetMemoryHints(const OpNode &node);

  void PrepinData(std::vector<tensor_data_store_queue_t> &tensor_to_store_queue,
                  const OpGraph &graph);

  void PresizeData(std::vector<tensor_data_store_queue_t> &tensor_to_store_queue,
                   const OpGraph &graph);

  void SetupOutputQueuesForGraph();

  // template <DALIOpType op_type>
  // // workspace_t<op_type> &GetWorkspace(QueueIdxs idxs, OpPartitionId partition_idx);
  // workspace_t<op_type> GetWorkspace(QueueIdxs idxs, const OpGraph &graph, OpPartitionId partition_idx);

  // template <DALIOpType op_type>
  // // workspace_t<op_type> &GetWorkspace(QueueIdxs idxs, const OpNode &node);
  // workspace_t<op_type> GetWorkspace(QueueIdxs idxs, const OpGraph &graph, const OpNode &node);

  // template <DALIOpType op_type>
  // void SetupInputOutput(workspace_t<op_type> &ws, const OpGraph &graph, const OpNode &node,
  //                       const std::vector<tensor_data_store_queue_t>& tensor_to_store_queue, const QueueIdxs idxs);

  // // template <DALIOpType op_type>
  // // void SetupPinned(workspace_t<op_type> &ws, const OpGraph &graph, const OpNode &node,
  // //                  const QueueIdxs idxs);

  // template <DALIOpType op_type>
  // void SetupStreamsAndEvents(workspace_t<op_type> &ws, const OpGraph &graph, const OpNode &node,
  //                            cudaStream_t mixed_op_stream, cudaStream_t gpu_op_stream,
  //                            const std::vector<std::vector<cudaEvent_t>> &mixed_op_events,
  //                            const QueueIdxs idxs);

  // template <DALIOpType op_type>
  // workspace_t<op_type> CreateWorkspace(const OpGraph &graph, const OpNode &node,
  //   const std::vector<tensor_data_store_queue_t> &tensor_to_store_queue, cudaStream_t mixed_op_stream,
  //   cudaStream_t gpu_op_stream, const std::vector<std::vector<cudaEvent_t>> &mixed_op_events,
  //                                              const QueueIdxs idxs);

  class EventList {
   public:
    inline EventList() {}
    inline EventList(int size, EventPool *event_pool) {
      DALI_ENFORCE(event_pool != nullptr);
      for (int i = 0; i < size; ++i) {
        events_.push_back(event_pool->GetEvent());
      }
    }

    inline cudaEvent_t GetEvent(int idx) { return events_[idx]; }

   private:
    vector<cudaEvent_t> events_;
  };

  int batch_size_, device_id_;
  size_t bytes_per_sample_hint_;
  // QueueSizes queue_depth_;
  int previous_gpu_queue_idx_ = -1;

  vector<string> output_names_;

  // Meta-data about our stage outputs for fast lookup
  std::vector<TensorNodeId> pipeline_outputs_;
  std::vector<EventList> gpu_output_events_;

  // Buffers are rotated between being 'free', where the
  // pipeline is ok to fill them with data, 'ready', where
  // they are already full of prepared data, and 'in-use',
  // where the user currently owns that buffer. A buffer
  // is marked as in-use when it is returned as and output.
  // The buffer is then returned the the ready queue the
  // next time Ouputs() is called.
  // std::queue<int> ready_queue_, free_queue_, in_use_queue_;
  // std::mutex ready_mutex_, free_mutex_;
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

  std::array<std::mutex, static_cast<int>(DALIOpType::COUNT)> stage_free_mutex_;
  std::array<std::mutex, static_cast<int>(DALIOpType::COUNT)> stage_ready_mutex_;
  std::array<std::condition_variable, static_cast<int>(DALIOpType::COUNT)> stage_free_cv_;
  std::array<std::condition_variable, static_cast<int>(DALIOpType::COUNT)> stage_ready_cv_;

  std::array<std::queue<int>, static_cast<int>(DALIOpType::COUNT)> stage_free_;
  std::array<std::queue<int>, static_cast<int>(DALIOpType::COUNT)> stage_ready_;
  std::array<int, static_cast<int>(DALIOpType::COUNT)> stage_queue_depths_;

  std::mutex ready_output_mutex_, in_use_mutex_;
  std::queue<OutputIdxs> ready_output_queue_;
  std::queue<OutputIdxs> in_use_queue_;

  // TODO Scoped acquire?
  QueueIdxs AcquireIdxs(DALIOpType stage);
  void ReleaseIdxs(DALIOpType stage, QueueIdxs idxs);

  void QueueOutputIdxs(QueueIdxs idxs);


  OpGraph *graph_ = nullptr;
  StreamPool stream_pool_;
  EventPool event_pool_;
  ThreadPool thread_pool_;
  std::vector<std::string> errors_;
  std::mutex errors_mutex_;
  bool exec_error_;
  ExecutorCallback cb_;
  QueueSizes queue_sizes_;
  std::vector<tensor_data_store_queue_t> tensor_to_store_queue_;
  cudaStream_t mixed_op_stream_, gpu_op_stream_;
  // MixedOpId -> queue_idx -> cudaEvent_t
  // To introduce dependency from MIXED to GPU Ops
  std::vector<std::vector<cudaEvent_t>> mixed_op_events_;
};

// #define USE_EXECUTOR_MEMBERS()            \
//  protected:                               \
//   using Executor::WorkspaceBlob;          \
//   using Executor::wss_;                   \
//   using Executor::batch_size_;            \
//   using Executor::device_id_;             \
//   using Executor::bytes_per_sample_hint_; \
//   using Executor::output_names_;          \
//   using Executor::gpu_output_events_;     \
//   using Executor::ready_cond_;            \
//   using Executor::graph_;                 \
//   using Executor::stream_pool_;           \
//   using Executor::event_pool_;            \
//   using Executor::thread_pool_

// template <DALIOpType op_type>
// // workspace_t<op_type> &Executor::GetWorkspace(QueueIdxs idxs, OpPartitionId partition_idx) {
// workspace_t<op_type> Executor::GetWorkspace(QueueIdxs idxs, const OpGraph &graph, OpPartitionId partition_idx) {
//   // auto &ws_vec = std::get<static_cast<size_t>(op_type)>(wss_[idxs[op_type]].op_data);
//   // return ws_vec[partition_idx];
//   return CreateWorkspace<op_type>(*graph_, graph_->Node(op_type, partition_idx), 
//       tensor_to_store_queue_, mixed_op_stream_, gpu_op_stream_, mixed_op_events_, idxs);
// }

// template <DALIOpType op_type>
// // workspace_t<op_type> &Executor::GetWorkspace(QueueIdxs idxs, const OpNode &node) {
// workspace_t<op_type> Executor::GetWorkspace(QueueIdxs idxs, const OpGraph &graph, const OpNode &node) {
//   DALI_ENFORCE(node.op_type == op_type,
//                "Wrong variant of method selected. DALIOpType does not match.");
//   // auto &ws_vec = std::get<static_cast<size_t>(op_type)>(wss_[idxs[op_type]].op_data);
//   // return ws_vec[node.partition_index];
//   return  CreateWorkspace<op_type>(*graph_, node, tensor_to_store_queue_, mixed_op_stream_,
//       gpu_op_stream_, mixed_op_events_, idxs);
// }



}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR_H_
