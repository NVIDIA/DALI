// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_TEST_OP_GRAPH_H_
#define DALI_TEST_OP_GRAPH_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <utility>

namespace dali {
namespace testing {

struct Node;

struct Edge {
  Edge() = default;
  Edge(Node *n, int index, bool explicit_GPU = false)
  : node(n), index(index), explicit_GPU(false) {}

  Node *node = nullptr;
  int index = 0;
  bool explicit_GPU = false;

  Edge gpu() const {
    auto ret = *this;
    ret.explicit_GPU = true;
    return ret;
  }
};

struct Node {
  Node() = default;
  Node(const Node &) = delete;
  Node(Node &&) = default;

  Node(std::string op_name, std::string node_name, int backend_mask)
  : op_name(std::move(op_name)), node_name(std::move(node_name)), backend_mask(backend_mask) {}

  template <typename... Inputs>
  Node &operator()(Inputs&&... inputs) {
    int junk[] = { (add_in(inputs), 0)... };
    (void)junk;
    return *this;
  }

  void add_in(Edge e) {
    in.push_back(e);
  }

  Edge operator[](int i) {
    if (i >= num_outputs)
      num_outputs = i+1;
    return { this, i };
  }

  enum Backend {
    CPU = 1,
    GPU = 2,
    AnyBackend = 3,
  };

  std::string op_name;
  std::string node_name;
  int backend_mask = 0;
  int num_outputs = 0;
  std::vector<Edge> in;
};

struct OpDAG {
  Node &add(std::string op_name, std::string node_name, int backend_mask = Node::AnyBackend) {
    Node node(std::move(op_name), std::move(node_name), backend_mask);
    auto result = nodes.emplace(node.op_name, std::move(node));
    assert(result.second && "Duplicate node");
    return result.first->second;
  }

  Node &add(std::string op_name, int backend_mask = Node::AnyBackend) {
    std::string node_name = op_name;
    for (int suffix = 0; nodes.count(node_name); suffix++) {
      node_name = op_name + std::to_string(suffix);
    }
    return add(std::move(op_name), std::move(node_name), backend_mask);
  }

  bool check_cycles(
      const Node *n,
      std::unordered_set<const Node *> &done,
      std::unordered_set<const Node *> &in_progress) const {
    if (done.count(n))
      return true;
    if (!in_progress.insert(n).second)
      return false;
    for (auto in : n->in)
      if (!check_cycles(in.node, done, in_progress))
        return false;
    done.insert(n);
    in_progress.erase(n);
    return true;
  }

  bool validate() const {
    std::unordered_set<const Node *> done;
    std::unordered_set<const Node *> in_progress;
    if (!check_cycles(&out, done, in_progress))
      return false;

    return in.num_outputs == 0 || done.count(&in);
  }

  std::unordered_map<std::string, Node> nodes;
  Node in{"", "__input", Node::AnyBackend};
  Node out{"", "__output", Node::AnyBackend};
};

/// @brief Creates a trivial graph with just one node.
/// @param op_name - name of the operator
/// @param backend_mask - backends to test
/// @param num_inputs  - number of inputs, if 0 then it's inferred from number
///                      of inputs supplied to a function running the graph
/// @param num_outputs - number of outputs, if 0 then it's inferred from number
///                      of reference outputs supplied to a function running the graph
OpDAG SingleOpGraph(const std::string &op_name,
                    int backend_mask = Node::AnyBackend,
                    int num_inputs = 0, int num_outputs = 0) {
  OpDAG dag;
  auto &n = dag.add(op_name, backend_mask);
  for (int i = 0; i < num_inputs; i++)
    n.in.push_back(dag.in[i]);
  for (int i = 0; i < num_outputs; i++)
    dag.out(n[i]);
  return dag;
}


}  // namespace testing
}  // namespace dali

#endif  // DALI_TEST_OP_GRAPH_H_
