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

#ifndef DALI_PIPELINE_EXECUTOR_QUEUE_METADATA_
#define DALI_PIPELINE_EXECUTOR_QUEUE_METADATA_

#include "dali/common.h"

namespace dali {

struct QueueIdxs {
  int &operator[](DALIOpType op_type) { return idxs[static_cast<size_t>(op_type)]; }

  const int &operator[](DALIOpType op_type) const { return idxs[static_cast<size_t>(op_type)]; }

  explicit QueueIdxs(int uniform_idx) : idxs{uniform_idx, uniform_idx, uniform_idx, uniform_idx} {}

 private:
  std::array<int, static_cast<size_t>(DALIOpType::COUNT)> idxs = {{0, 0, 0, 0}};
};

struct QueueSizes {
  QueueSizes() = default;
  QueueSizes(int output_size) : cpu_size(1), mixed_size(output_size), gpu_size(output_size) {}
  QueueSizes(int cpu_size, int mixed_size, int gpu_size)
      : cpu_size(cpu_size), mixed_size(mixed_size), gpu_size(gpu_size) {}

  int cpu_size = 1, mixed_size = 1, gpu_size = 1;
};

struct OutputIdxs {
  int mixed;
  int gpu;

  int &operator[](DALIOpType op_type) {
    if (op_type == DALIOpType::MIXED) {
      return mixed;
    } else {
      return gpu;
    }
  }

  const int &operator[](DALIOpType op_type) const {
    if (op_type == DALIOpType::MIXED) {
      return mixed;
    } else {
      return gpu;
    }
  }
};

static std::ostream &operator<<(std::ostream &os, QueueIdxs idxs) {
  os << "{" << idxs[DALIOpType::SUPPORT] << ", " << idxs[DALIOpType::CPU] << ", "
     << idxs[DALIOpType::MIXED] << ", " << idxs[DALIOpType::GPU] << "}";
  return os;
}


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


}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_QUEUE_METADATA_