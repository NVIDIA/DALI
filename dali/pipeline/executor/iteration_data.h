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

#ifndef DALI_ITERATION_DATA_H
#define DALI_ITERATION_DATA_H

#include <unordered_map>
#include <memory>

namespace dali {

/// Maps operator name to the Operator Traces
using operator_trace_map_t = std::unordered_map<
        std::string /* op_name */,
        std::unordered_map<std::string /* trace_name */, std::string /* trace_value */>
>;

struct IterationData {
  std::shared_ptr<operator_trace_map_t> operator_traces;
};

}

#endif //DALI_ITERATION_DATA_H
