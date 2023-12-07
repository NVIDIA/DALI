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

#ifndef DALI_PIPELINE_EXECUTOR2_TF_EXEC_GRAPH_H_
#define DALI_PIPELINE_EXECUTOR2_TF_EXEC_GRAPH_H_

#include <cassert>
#include <memory>
#include <variant>

#include "../graph.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/core/cuda_event_pool.h"

#include "third_party/taskflow/taskflow/taskflow.hpp"  // TODO(michalz): Add it to cmake

namespace dali {
namespace exec2 {

struct ExecNode;

class CUDAEventLease : public CUDAEvent {
 public:
  CUDAEventLease() = default;
  explicit CUDAEventLease(CUDAEventPool &pool, int device_id = -1) : owner_(&pool) {
    *static_cast<CUDAEvent *>(this) = pool.Get(device_id);
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

class ExecNode;

struct ExecEdge {
  ExecNode *producer = nullptr;
  ExecNode *consumer = nullptr;

  int producer_output_idx = 0;
  int consumer_input_idx = 0;
};

class ExecNode {
 public:
  OperatorNode *op_node_ = nullptr;

  std::vector<ExecEdge *> inputs_, outputs_;

  OperatorBase *op_instance = nullptr;

  Workspace &current_ws() { return ws_[ws_idx_]; }
  void switch_ws() { ws_idx_ = 1 - ws_idx_; }

 private:
  Workspace ws_[2];
  int ws_idx_ = 0;


};

struct ExecGraph {
  std::list<ExecEdge> edges;
  std::list<ExecNode> nodes;
};

}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR2_EXEC_TF_GRAPH_H_
