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

#ifndef DALI_PIPELINE_EXECUTOR_ITERATION_DATA_H_
#define DALI_PIPELINE_EXECUTOR_ITERATION_DATA_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "dali/pipeline/operator/checkpointing/checkpoint.h"

namespace dali {

/**
 * Operator Traces is a mechanism, where an Operator can provide any arbitrary information to
 * the end user. Under `trace_name` key, the Operator assigns `trace_value` as the information
 * mentioned above. Using the provided API, user will be able to retrieve this information after
 * the iteration ends - at the same time when he's able to access outputs from the pipeline.
 *
 * @see daliGetOperatorTrace, daliGetNumOperatorTraces
 *
 * Here are few examples of these traces, but essentially sky is the limit:
 *   - "execution_time" -> "432 sec"
 *   - "number_of_unprocessed_samples" -> "100"
 *   - "next_batch_ready" -> "true"
 */
using operator_trace_map_t = std::unordered_map<
        std::string /* op_name */,
        std::unordered_map<std::string /* trace_name */, std::string /* trace_value */>
>;


/**
 * Contains the data of an iteration. This data is shared across all Workspaces, that belong to
 * a single iteration.
 */
struct IterationData {
  std::shared_ptr<operator_trace_map_t> operator_traces;
  Checkpoint checkpoint;
};

}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_ITERATION_DATA_H_
