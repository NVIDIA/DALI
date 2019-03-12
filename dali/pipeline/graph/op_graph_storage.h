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

#ifndef DALI_PIPELINE_GRAPH_OP_GRAPH_STORAGE_H_
#define DALI_PIPELINE_GRAPH_OP_GRAPH_STORAGE_H_

#include <vector>

#include "dali/pipeline/graph/op_graph.h"
#include "dali/pipeline/util/event_pool.h"
#include "dali/pipeline/workspace/workspace_data_factory.h"

namespace dali {

DLL_PUBLIC std::vector<tensor_data_store_queue_t> CreateBackingStorageForTensorNodes(
    const OpGraph& op_graph, int batch_size, const std::vector<int>& queue_sizes);

// Mapping from MixedOp partition id to queue of corresponding events
DLL_PUBLIC std::vector<std::vector<cudaEvent_t>> CreateEventsForMixedOps(EventPool& event_pool,
                                                                         const OpGraph& op_graph,
                                                                         int queue_depth);

}  // namespace dali

#endif  // DALI_PIPELINE_GRAPH_OP_GRAPH_STORAGE_H_
