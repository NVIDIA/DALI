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

#include <memory>

#include "graph.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/core/cuda_event.h"

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
      owner_.Put(std::move(*static_cast<CUDAEvent *>(this)));
      owner_ = nullptr;
    }
  }

  CUDAEventLease &operator=(const CUDAEventLease &other) {
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
    std::shared_ptr<TensorList<CPUBackend>> cpu_data;
    std::shared_ptr<TensorList<GPUBackend>> gpu_data;
    CUDAEventLease cpu_ready, gpu_ready;
  };
  std::deque<OutputQueueEntry> queue;
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
