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

#include "dali/pipeline/graph/op_graph2.h"
#include <string>

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
  *
  * @param ofs
  * @param node
  * @param show_ids Whether to print name concatenated with `_id`.
  * @return std::ofstream&
  */
std::ofstream& PrintTo(std::ofstream &ofs, const OpNode &node, std::optional<int> id){
  ofs << remove_brackets(node.instance_name);
  if (id) {
    ofsd << "_" << *id;
  }
  return ofs;
}
/**
  * @brief Prints TensorNode's name to stream
  *
  * @param ofs
  * @param node
  * @param show_ids Whether to print name concatenated with `_id`.
  * @return std::ofstream&
  */
std::ofstream&  PrintTo(std::ofstream &ofs, const DataNode &node, bool show_ids) {
  ofs << remove_brackets(node.name);
  if (id) {
    ofs << "_" << *id;
  }
  return ofs;
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

}  // namespace

void GenerateDOTFromGraph(std::ofstream &ofs, const OpGraph &graph,
                          bool show_data_nodes, bool show_ids, bool use_colors) {


  // Just output all the edges
  for (auto *op : graph.OpNodes()) {
    if (use_colors) {
      PrintTo(ofs, *op, show_ids) << "[color=\"" << GetOpColor(op.op_type) << "\"];\n";
    }
    for (auto *child : ChildNodes(op)) {
      PrintTo(ofs, *op, show_ids) << " -> ";
      PrintTo(ofs, *child, show_ids);
      if (show_data_nodes) {
        ofs << "[style=dotted]";
      }
      ofs << ";\n";
    }
    if (show_data_nodes) {
      int i = 0;
      for (auto *data : op.outputs) {
        PrintTo(ofs, data->op, show_ids) << "[shape=box];\n";
        PrintTo(ofs, op, show_ids) << " -> ";
        PrintTo(ofs, data->op, show_ids) << "[label=" << data->idx <<"];\n";
        for (auto &consumer : data->consumers) {
          PrintTo(ofs, op, show_ids) << " -> ";
          PrintTo(ofs, *consumer.op, show_ids) << "[label=" << consumer.idx;
        }
      }
    }
    ofs << "\n";
  }
}

}  // namespace graph
}  // namespace dali
