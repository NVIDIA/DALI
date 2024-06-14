// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include "dali/pipeline/graph/op_graph2.h"
#include "dali/pipeline/graph/graph2dot.h"

namespace dali {
namespace graph {

namespace {

std::string remove_brackets(std::string input) {
  // We have output indexing via the `[idx]` syntax, replace the brackets with something
  // allowed in dot
  std::replace(input.begin(), input.end(), '[', '_');
  std::replace(input.begin(), input.end(), ']', '_');
  return input;
}

/**
  * @brief Prints instance_name of OpNode to stream
  */
std::ostream& PrintTo(std::ostream &os, const OpNode &node) {
  os << remove_brackets(node.instance_name);
  return os;
}
/**
  * @brief Prints TensorNode's name to stream
  * @return std::ostream&
  */
std::ostream&  PrintTo(std::ostream &os, const DataNode &node) {
  os << remove_brackets(node.name);
  return os;
}
std::string GetOpColor(OpType op_type) {
  switch (op_type) {
    case OpType::CPU:
      return "blue";
    case OpType::GPU:
      return "#76b900";
    case OpType::MIXED:
      return "cyan";
    default:
      return "black";
  }
}

std::vector<OpNode *> ChildNodes(const OpNode &op) {
  std::unordered_set<OpNode *> seen;
  std::vector<OpNode *> children;
  for (auto *out : op.outputs) {
    for (auto edge : out->consumers)
      if (edge.op)
        if (seen.insert(edge.op).second)
          children.push_back(edge.op);
  }
  return children;
}

}  // namespace

void GenerateDOTFromGraph(std::ostream &os, const OpGraph &graph,
                          bool show_data_nodes, bool use_colors) {
  os << "digraph graphname {\n";

  for (auto &op : graph.OpNodes()) {
    if (use_colors) {
      PrintTo(os, op) << "[color=\"" << GetOpColor(op.op_type) << "\"];\n";
    }
    for (auto *child : ChildNodes(op)) {
      PrintTo(os, op) << " -> ";
      PrintTo(os, *child);
      if (show_data_nodes) {
        os << "[style=dotted]";
      }
      os << ";\n";
    }
    if (show_data_nodes) {
      int i = 0;
      for (auto *data : op.outputs) {
        PrintTo(os, *data) << "[shape=box];\n";
        PrintTo(os, op) << " -> ";
        PrintTo(os, *data) << "[label=" << i++ << "];\n";
        for (auto &consumer : data->consumers) {
          PrintTo(os, *data) << " -> ";
          PrintTo(os, *consumer.op) << "[label=" << consumer.idx << "];\n";
        }
      }
    }
    os << "\n";
  }

  os << "}\n";
}

}  // namespace graph
}  // namespace dali
