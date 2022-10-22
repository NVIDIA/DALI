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

#include <set>
#include <string>
#include <vector>

#include "dali/pipeline/graph/op_graph_storage.h"

namespace dali {

std::vector<tensor_data_store_queue_t> CreateBackingStorageForTensorNodes(
    const OpGraph &op_graph, int batch_size, const std::vector<int> &queue_sizes,
    const std::vector<std::string> &output_names) {
  DALI_ENFORCE(static_cast<int>(queue_sizes.size()) == op_graph.NumTensor(),
               "Data queue sizes undefined for some Tensor nodes.");
  std::vector<tensor_data_store_queue_t> result;
  result.resize(op_graph.NumTensor());

  std::set<int64_t> outputs;
  auto output_ids = op_graph.GetOutputs(output_names);;
  outputs.insert(output_ids.begin(), output_ids.end());

  // Assign data to each Tensor node in graph
  for (int i = 0; i < op_graph.NumTensor(); i++) {
    const auto &tensor = op_graph.Tensor(i);
    auto producer_op_type = op_graph.Node(tensor.producer.node).op_type;
    result[i] =
        BatchFactory(producer_op_type, tensor.producer.storage_device, batch_size, queue_sizes[i]);

    bool is_output = outputs.count(tensor.id) > 0;
    tuple_for_each(result[i], [&](auto &x) {
      x.num_consumers = tensor.consumers.size();
      if (is_output)
        x.num_consumers++;
    });
  }
  return result;
}

MixedOpEventMap CreateEventsForMixedOps(EventPool &event_pool, const OpGraph &op_graph,
                                        int mixed_queue_depth) {
  MixedOpEventMap result;
  result.resize(op_graph.NumOp(OpType::MIXED));
  for (int i = 0; i < op_graph.NumOp(OpType::MIXED); i++) {
    result[i].resize(mixed_queue_depth);
    for (int j = 0; j < mixed_queue_depth; j++) {
      result[i][j] = event_pool.GetEvent();
    }
  }
  return result;
}

}  // namespace dali
