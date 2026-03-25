// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <unordered_map>
#include <string>
#include "dali/pipeline/graph/node_meta.h"
#include "dali/pipeline/graph/graph_util.h"

namespace dali {
namespace graph {

namespace {

void PropagateDataNodeMetadata(OpNode &node) {
  if (node.visited)
    return;
  Visit<OpNode> visit(&node);

  std::unordered_map<std::string, OpSpec::InOutDesc> input_descs;

  for (auto *in : node.inputs) {
    if (in->producer.op) {
      PropagateDataNodeMetadata(*in->producer.op);
      auto &prod_spec = in->producer.op->spec;
      input_descs[prod_spec.OutputName(in->producer.idx)] = prod_spec.OutputDesc(in->producer.idx);
    }
  }

  for (int i = 0; i < node.spec.NumInput(); i++) {
    const auto &name = node.spec.InputName(i);
    auto it = input_descs.find(name);
    if (it == input_descs.end()) {
      assert(!"Internal error when computing node metadata.");
      continue;
    }
    auto &desc = node.spec.MutableInputDesc(i);
    desc.ndim = it->second.ndim;
    desc.dtype = it->second.dtype;
    desc.layout = it->second.layout;
  }

  node.spec.InferOutputMetadata();
}

}  // namespace

void ComputeDataNodeMetadata(OpGraph &graph) {
  ClearVisitMarkers(graph.OpNodes());
  for (auto &node : graph.OpNodes())
    PropagateDataNodeMetadata(node);
}

}  // namespace graph
}  // namespace dali
