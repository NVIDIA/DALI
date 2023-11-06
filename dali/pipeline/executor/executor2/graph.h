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

#ifndef DALI_PIPELINE_EXECUTOR2_GRAPH_H_
#define DALI_PIPELINE_EXECUTOR2_GRAPH_H_

#include "dali/pipeline/operator/op_spec.h"
#include <map>
#include <unordered_map>
#include <vector>
#include <string_view>

namespace dali {
namespace exec2 {

class Graph;

class GraphBuilder {
 public:
  GraphBuilder();
  ~GraphBuilder();

  void Add(const OpSpec &spec);
  void Build(Graph &graph);
 private:
  friend class Graph;
  class GraphBuilderImpl;
  std::unique_ptr<GraphBuilderImpl> impl;
};

struct OperatorNode;

struct DataNode {
  std::string_view name;
  OperatorNode *producer = nullptr;
  std::vector<OperatorNode *> consumers;
  StorageDevice backend;
};

struct OperatorNode {
  std::string_view name;
  std::vector<DataNode *> inputs;
  std::vector<DataNode *> outputs;

  DataNode *Input(int idx) const {
    assert(static_cast<size_t>(idx) < inputs.size());
    return inputs[idx];
  }

  DataNode *Output(int idx) const {
    assert(static_cast<size_t>(idx) < outputs.size());
    return outputs[idx];
  }

  DataNode *ArgInput(const std::string &name) const {
    auto &args = spec.ArgumentInputs();
    auto it = args.find(name);
    if (it == args.end())
      throw std::invalid_argument(make_string("No such argument input: ", name));
    return inputs[it->second];
  }

  DataNode *output(int idx) const {
    return outputs[idx];
  }

  OpSpec spec;
  OpType backend;
};

class Graph {
 public:
  template <typename Key>
  OperatorNode &op_node(const Key &key) {
    auto it = op_nodes.find(key);
    if (it == op_nodes.end())
      throw std::invalid_argument(make_string("No such operator node: ", key));
    return it->second;
  }

  template <typename Key>
  DataNode &data_node(const Key &key) {
    auto it = data_nodes.find(key);
    if (it == data_nodes.end())
      throw std::invalid_argument(make_string("No such data node: ", key));
    return it->second;
  }

 private:
  friend class GraphBuilder;
  friend class GraphBuilder::GraphBuilderImpl;
  std::map<std::string, OperatorNode> op_nodes;
  std::map<std::string, DataNode> data_nodes;
  std::vector<DataNode *> inputs, outputs;

  void Validata() const;
};

}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR2_GRAPH_H_

