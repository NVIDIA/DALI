// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_EXECUTOR_QUEUE_METADATA_H_
#define DALI_PIPELINE_EXECUTOR_QUEUE_METADATA_H_

#include "dali/common.h"

namespace dali {

// Used to store Stage queue sizes
struct StageQueues {
  int &operator[](OpType op_type) { return idxs[static_cast<size_t>(op_type)]; }

  const int &operator[](OpType op_type) const { return idxs[static_cast<size_t>(op_type)]; }

  StageQueues() = default;

  explicit StageQueues(int uniform_idx)
      : idxs{uniform_idx, uniform_idx, uniform_idx, uniform_idx} {}

  StageQueues(int support, int cpu, int mixed, int gpu) {
    operator[](OpType::SUPPORT) = support;
    operator[](OpType::CPU) = cpu;
    operator[](OpType::MIXED) = mixed;
    operator[](OpType::GPU) = gpu;
  }

 private:
  std::array<int, static_cast<size_t>(OpType::COUNT)> idxs = {{0, 0, 0, 0}};
};

// Used for indexing into stage queues
using QueueIdxs = StageQueues;

struct QueueSizes {
  QueueSizes() = default;
  explicit QueueSizes(int output_size)
      : cpu_size(output_size), gpu_size(output_size) {}
  QueueSizes(int cpu_size, int gpu_size)
      : cpu_size(cpu_size), gpu_size(gpu_size) {}

  int cpu_size = 1, gpu_size = 1;
};

struct OutputIdxs {
  explicit OutputIdxs(int queue_idx) : mixed(queue_idx), gpu(queue_idx) {}
  OutputIdxs(int mixed, int gpu) : mixed(mixed), gpu(gpu) {}

  int mixed;
  int gpu;

  int &operator[](OpType op_type) {
    if (op_type == OpType::MIXED) {
      return mixed;
    } else {
      return gpu;
    }
  }

  const int &operator[](OpType op_type) const {
    if (op_type == OpType::MIXED) {
      return mixed;
    } else {
      return gpu;
    }
  }
};

static std::ostream &operator<<(std::ostream &os, StageQueues idxs) {
  os << "{" << idxs[OpType::SUPPORT] << ", " << idxs[OpType::CPU] << ", "
     << idxs[OpType::MIXED] << ", " << idxs[OpType::GPU] << "}";
  return os;
}


static OpType PreviousStage(OpType op) {
  switch (op) {
    case OpType::CPU:
      return OpType::SUPPORT;
    case OpType::MIXED:
      return OpType::CPU;
    case OpType::GPU:
      return OpType::MIXED;
    default:
      return static_cast<OpType>(-1);  // No previous OpType
  }
}

static bool HasPreviousStage(OpType op) {
  return op != OpType::SUPPORT;
}

static OpType NextStage(OpType op) {
  switch (op) {
    case OpType::SUPPORT:
      return OpType::CPU;
    case OpType::CPU:
      return OpType::MIXED;
    case OpType::MIXED:
      return OpType::GPU;
    default:
      return static_cast<OpType>(-1);  // No next OpType
  }
}

static bool HasNextStage(OpType op) {
  return op != OpType::GPU;
}


}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_QUEUE_METADATA_H_
