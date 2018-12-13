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
#include <iostream>

namespace dali {
namespace testing {

struct Node;

struct NodeOutput {
  NodeOutput(Node *n, int index, bool explicit_GPU = false) : node(n), index(index), explicit_GPU(false) {}

  Node *node = nullptr;
  int index = 0;
  bool explicit_GPU = false;

  NodeOutput gpu() const {
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

  NodeOutput operator()(int i) { return { this, i }; }

  enum Backend {
    CPU = 1,
    GPU = 2,
    AnyBackend = 3,
  };

  std::string op_name;
  std::string node_name;
  int backend_mask = 0;
  std::vector<NodeOutput> inputs;
};

struct NodePtr {
  NodePtr(Node *ptr = nullptr) : ptr(ptr) {}  // NOLINT
  Node &operator*() const { return *ptr; }
  Node *operator->() const { return ptr; }
  NodeOutput operator()(int output_index) const { return (*ptr)(output_index); }
  Node *ptr;
};


struct NodeInputAssignment {
  Node *ptr;
};

inline NodeInputAssignment operator,(NodeInputAssignment n, NodeOutput o) {
  n.ptr->inputs.push_back(o);
  return n;
}

inline NodeInputAssignment operator<<(Node &node, NodeOutput o) {
  node.inputs.push_back(o);
  return { &node };
}

template <typename T>
NodeInputAssignment operator<<(NodePtr ptr, T&& rhs) {
  return *ptr << std::forward<T>(rhs);
}



struct OpDAG {
  OpDAG() = default;
  OpDAG(const std::string &op_name, int backend_mask = Node::AnyBackend, int num_inputs = 1, int num_outputs = 1) {
    auto n = add(op_name);
    for (int i = 0; i < num_inputs; i++)
      n << input(i);
    for (int i = 1; i < num_outputs; i++)
      output << n(0);
  }

  NodePtr add(std::string op_name, int backend_mask = Node::AnyBackend) {
    std::string node_name = op_name;
    for (int suffix = 0; nodes.count(node_name); suffix++) {
      node_name = op_name + std::to_string(suffix);
    }
    Node node(std::move(op_name), std::move(node_name), backend_mask);
    return &nodes.emplace(node.op_name, std::move(node)).first->second;
  }

  bool check_cycles(
      const Node *n,
      std::unordered_set<const Node *> &done,
      std::unordered_set<const Node *> &in_progress) const {
    if (done.count(n))
      return true;
    if (!in_progress.insert(n).second)
      return false;
    for (auto in : n->inputs)
      if (!check_cycles(in.node, done, in_progress))
        return false;
    done.insert(n);
    in_progress.erase(n);
    return true;
  }

  bool validate(bool checkInputReferenced) const {
    std::unordered_set<const Node *> done;
    std::unordered_set<const Node *> in_progress;
    if (!check_cycles(&output, done, in_progress))
      return false;
    if (checkInputReferenced)
      return done.count(&input);
    return true;
  }

  std::unordered_map<std::string, Node> nodes;
  Node input{"", "__input", Node::AnyBackend}, output{"output", "__output", Node::AnyBackend};
};

}  // testing
}  // dali

#endif
