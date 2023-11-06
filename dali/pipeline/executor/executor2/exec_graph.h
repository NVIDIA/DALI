// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_EXECUTOR2_EXEC_GRAPH_H_
#define DALI_PIPELINE_EXECUTOR2_EXEC_GRAPH_H_

#include <cassert>
#include <memory>
#include <variant>

#include "graph.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/core/cuda_event_pool.h"

namespace dali {
namespace exec2 {

struct ExecNode;

class CUDAEventLease : public CUDAEvent {
 public:
  CUDAEventLease() = default;
  CUDAEventLease(CUDAEventPool &pool) : owner_(&pool) {
    *static_cast<CUDAEvent *>(this) = pool.Get();
  }

  ~CUDAEventLease() {
    reset();
  }

  void reset() {
    if (*this) {
      assert(owner_);
      owner_->Put(std::move(*static_cast<CUDAEvent *>(this)));
      owner_ = nullptr;
    }
  }

  CUDAEventLease &operator=(CUDAEventLease &&other) {
    reset();
    CUDAEvent::reset(other.release());
    owner_ = other.owner_;
    other.owner_ = nullptr;
    return *this;
  }

 private:
  CUDAEventPool *owner_;
};

struct ExecDataNode {
  template <typename Backend>
  std::shared_ptr<TensorList<Backend>> get(AccessOrder order) {
    return get(order, Backend());
  }

  std::shared_ptr<TensorList<CPUBackend>> get(AccessOrder order, CPUBackend) {
    if (cpu_ready)
        order.wait(cpu_ready);
    return cpu_data;
  }

  std::shared_ptr<TensorList<GPUBackend>> get(AccessOrder order, GPUBackend) {
    if (gpu_ready)
        order.wait(gpu_ready);
    return gpu_data;
  }

  struct OutputQueueEntry {
    std::shared_ptr<TensorList<CPUBackend>> cpu;
    std::shared_ptr<TensorList<GPUBackend>> gpu;

    CUDAEventLease ready;
  };

  std::mutex mtx;
  std::condition_variable read, write;

  template <typename Backend>
  void push(std::shared_ptr<TensorList<Backend>> tl, AccessOrder writeOrder)
  {
    static_assert(std::is_same_v<Backend, CPUBackend> || std::is_same_v<Backend, GPUBackend>);

    std::lock_guard g(mtx);
    if constexpr (std::is_same_v<Backend, GPUBackend>)
      queue.emplace(CUDAEventPool::instance().Get()

  }

  int max_queue_depth = 1;

  std::queue<OutputQueueEntry> queue;

  ExecNode *producer = nullptr;
};

struct ExecNode {
  OperatorNode *op_node = nullptr;
  OperatorBase *op_instance = nullptr;
  std::vector<ExecDataNode *> inputs, outputs;
};

struct ExecGraph {
  std::vector<std::unique_ptr<ExecDataNode>> data_nodes;
  std::vector<std::unique_ptr<ExecNode>> op_nodes;
};

}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR2_EXEC_GRAPH_H_
