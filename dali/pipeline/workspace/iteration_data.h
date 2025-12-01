// Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_WORKSPACE_ITERATION_DATA_H_
#define DALI_PIPELINE_WORKSPACE_ITERATION_DATA_H_

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

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
using operator_trace_map_t = std::map<
  std::string /* trace_name */, std::string /* trace_value */, std::less<>>;

class Checkpoint;

class OperatorTraces {
 public:
  /** Gets operator traces for a single operator.
   *
   * This function is thread safe, however, the map returned by it is not.
   * The operator must ensure that it does not attempt to modify the map from multiple threads.
   */
  operator_trace_map_t &Get(std::string_view operator_name) {
    std::lock_guard g(mtx_);
    // TODO(michalz): in C++26, we can just use map_[operator_name] again, but until that - no luck
    auto it = map_.find(operator_name);
    if (it == map_.end()) {
      bool inserted;
      std::tie(it, inserted) = map_.emplace(std::string(operator_name), operator_trace_map_t{});
      assert(inserted);
    }
    return it->second;
  }

  /** Obtains a copy of trace maps for all operators.
   *
   * This function is thread safe.
   */
  auto GetCopy() const {
    std::lock_guard g(mtx_);
    return map_;
  }

  /** Gets a reference to the operator trace map */
  auto &GetRef() const & {
    return map_;
  }

 private:
  mutable std::mutex mtx_;
  std::map<
        std::string /* op_name */,
        operator_trace_map_t /* per-operator traces */,
        std::less<>
  > map_;
};

/**
 * Contains the data of an iteration. This data is shared across all Workspaces that belong to
 * a single iteration.
 */
struct IterationData {
  /** The index of the current iteration. */
  int64_t iteration_index = 0;

  /** Default batch size for the current iteration.
   *
   * Presently this is the batch size set by external sources or the maximum batch size,
   * if no external source is present.
   * Actual batch size may change, e.g. due to conditional execution.
   */
  int default_batch_size = 0;

  OperatorTraces operator_traces;
  std::shared_ptr<Checkpoint> checkpoint;
};

using SharedIterData = std::shared_ptr<IterationData>;

}  // namespace dali

#endif  // DALI_PIPELINE_WORKSPACE_ITERATION_DATA_H_

