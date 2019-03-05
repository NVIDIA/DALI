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

#include <vector>

#include "dali/pipeline/graph/op_graph_storage.h"

namespace dali {

std::vector<tensor_data_store_t> CreateBackingStorageForTensorNodes(const OpGraph &op_graph,
                                                                    int batch_size) {
  std::vector<tensor_data_store_t> result;
  result.resize(op_graph.NumTensor());
  // Assign data to each Tensor node in graph
  for (int i = 0; i < op_graph.NumTensor(); i++) {
    const auto &tensor = op_graph.Tensor(i);
    auto producer_op_type = op_graph.Node(tensor.producer.node).op_type;
    result[i] = BatchFactory(producer_op_type, tensor.producer.storage_device, batch_size);
  }
  return result;
}

std::vector<cudaEvent_t> CreateEventsForMixedOps(EventPool &event_pool, const OpGraph &op_graph) {
  std::vector<cudaEvent_t> result;
  result.resize(op_graph.NumOp(OpType::MIXED));
  for (int i = 0; i < op_graph.NumOp(OpType::MIXED); i++) {
    result[i] = event_pool.GetEvent();
  }
  return result;
}

}  // namespace dali
