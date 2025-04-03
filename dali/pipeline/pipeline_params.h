// Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_PIPELINE_PARAMS_H_
#define DALI_PIPELINE_PIPELINE_PARAMS_H_

#include <optional>
#include "dali/pipeline/executor/queue_metadata.h"
#include "dali/pipeline/executor/executor_type.h"

namespace dali {

/** Pipeline parameters used on construction of the Pipeline object */
struct PipelineParams {
  std::optional<int> max_batch_size;
  std::optional<int> num_threads;
  std::optional<int> device_id;
  std::optional<int64_t> seed;
  std::optional<ExecutorType> executor_type;
  std::optional<ExecutorFlags> executor_flags;
  std::optional<QueueSizes> prefetch_queue_depths;
  std::optional<bool> enable_checkpointing;
  std::optional<bool> enable_memory_stats;
  std::optional<size_t> bytes_per_sample_hint;

  PipelineParams& Update(const PipelineParams &p) {
    #define UPDATE_IF_SET(field) if (p.field.has_value()) field = p.field;
    UPDATE_IF_SET(max_batch_size);
    UPDATE_IF_SET(num_threads);
    UPDATE_IF_SET(device_id);
    UPDATE_IF_SET(seed);
    UPDATE_IF_SET(executor_type);
    UPDATE_IF_SET(executor_flags);
    UPDATE_IF_SET(prefetch_queue_depths);
    UPDATE_IF_SET(enable_checkpointing);
    UPDATE_IF_SET(enable_memory_stats);
    UPDATE_IF_SET(bytes_per_sample_hint);
    #undef UPDATE_IF_SET
    return *this;
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_PIPELINE_PARAMS_H_
