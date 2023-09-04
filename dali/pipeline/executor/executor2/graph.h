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

namespace dali {
namespace exec2 {

class Graph;

class GraphBuilder {
 public:
  void Add(const OpSpec &spec);
  void Build(Graph &graph);
};

struct OperatorNode;

struct DataNode {
  std::string name;
  OperatorNode *producer = nullptr;
  std::vector<OperatorNode *> consumers;
  StorageDevice backend;
};

struct OperatorNode {
  std::vector<DataNode *> inputs;
  std::vector<DataNode *> outputs;
  OpSpec spec;
  OpType backend;
};

class Graph {
 public:


 private:
  std::map<std::string, OperatorNode> op_nodes;
  std::map<std::string, DataNode> data_nodes;
  std::vector<DataNode *> inputs, outputs;
};

}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR2_GRAPH_H_

