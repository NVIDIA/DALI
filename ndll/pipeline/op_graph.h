// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OP_GRAPH_H_
#define NDLL_PIPELINE_OP_GRAPH_H_

#include <map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <fstream>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/operator.h"

namespace ndll {

using OpPtr = unique_ptr<Operator>;

typedef int64 NodeID;

struct OpNode {
  inline OpNode() {}
  virtual ~OpNode() = default;
  OpNode& operator=(const OpNode&) = delete;

  OpNode(OpNode &&) = default;
  OpNode& operator=(OpNode &&) = default;

  OpPtr op;
  NodeID id;
  OpSpec spec;
  std::set<NodeID> parents, children;
  std::string instance_name;
};

// Stores meta-data about a tensor and how it
// is used by a producer/consumer node.
struct TensorMeta {
  NodeID node;
  Index index;
  bool is_cpu;
};

/**
 * @brief Stores all meta-data about a graph of operations to be run
 * keeps track of useful meta-data about consumers/producers of
 * different intermediates.
 *
 * Operators in the graph have a global NodeID that is assigned in
 * the order ops are added to the graph. Operators also have an
 * index within the set of ops of its type (cpu, mixed, gpu).
 * This enables us to iterate over select portions of the graph, or
 * the entire graph.
 *
 * TODO(tgale): The Executor uses alot of information about the graph
 * that is not that easily accessed through the graph (e.g., we often
 * go into the OpSpec of the node directly). Now that we have running
 * executors, consider refactoring this API to make important info
 * more obviously available.
 */
class OpGraph {
 public:
  inline OpGraph() {}
  inline ~OpGraph() = default;

  /**
   * @brief Adds an op with the input specification to the graph.
   */
  void AddOp(const OpSpec &spec, const std::string& name);

  /**
   * @brief Removes the node with the specified NodeID from
   * the graph. Fails if the removal would produce an invalid
   * graph.
   */
  void RemoveOp(NodeID id);

  /**
   * @brief Returns the total number of ops in the graph.
   */
  inline Index NumOp() const {
    return NumCPUOp() + NumGPUOp() + NumMixedOp();
  }

  /**
   * @brief Returns the number of cpu ops in the graph.
   */
  inline Index NumCPUOp() const { return cpu_nodes_.size(); }

  /**
   * @brief Returns the number of gpu ops in the graph.
   */
  inline Index NumGPUOp() const { return gpu_nodes_.size(); }

  /**
   * @brief Returns the number of mixed ops in the graph.
   */
  inline Index NumMixedOp() const { return mixed_nodes_.size(); }

  /**
   * @brief Returns a reference to the `idx`-th cpu op that was
   * added to the graph.
   */
  inline Operator& cpu_op(Index idx) {
    NDLL_ENFORCE_VALID_INDEX(idx, (Index)cpu_nodes_.size());
    return *cpu_nodes_[idx].op;
  }

  /**
   * @brief Returns the node object for the `idx`-th cpu op that
   * was added to the graph.
   */
  inline OpNode& cpu_node(Index idx) {
    NDLL_ENFORCE_VALID_INDEX(idx, (Index)cpu_nodes_.size());
    return cpu_nodes_[idx];
  }

  /**
   * @brief Returns a reference to the `idx`-th gpu op that
   * was added to the graph.
   */
  inline Operator& gpu_op(Index idx) {
    NDLL_ENFORCE_VALID_INDEX(idx, (Index)gpu_nodes_.size());
    return *gpu_nodes_[idx].op;
  }

  /**
   * @brief Returns the node object for the `idx`-th gpu op that
   * was added to the graph.
   */
  inline OpNode& gpu_node(Index idx) {
    NDLL_ENFORCE_VALID_INDEX(idx, (Index)gpu_nodes_.size());
    return gpu_nodes_[idx];
  }

  /**
   * @brief Returns a reference to the `idx`-th mixed op
   * that was added to the graph.
   */
  inline Operator& mixed_op(Index idx) {
    NDLL_ENFORCE_VALID_INDEX(idx, (Index)mixed_nodes_.size());
    return *mixed_nodes_[idx].op;
  }

  /**
   * @brief Returns the node object for the `idx`-th mixed op that
   * was added to the graph.
   */
  inline OpNode& mixed_node(Index idx) {
    NDLL_ENFORCE_VALID_INDEX(idx, (Index)mixed_nodes_.size());
    return mixed_nodes_[idx];
  }

  /**
   * @brief Returns the graph node with the given name.
   * This function is much slower than the version taking
   * index as argument so should not be used in performance
   * critical section of the code.
   */
  OpNode& node(const std::string& name);

  /**
   * @brief Returns the graph node with the given index in the graph.
   */
  OpNode& node(NodeID id);

  /**
   * @brief Returns the type (cpu, gpu, mixed) of the node
   * at the given index.
   */
  inline NDLLOpType NodeType(NodeID id) const {
    NDLL_ENFORCE_VALID_INDEX(id, (Index)id_to_node_map_.size());
    return id_to_node_map_[id].first;
  }

  /**
   * @brief Returns the index of the node with the specified id
   * among nodes of its type.
   */
  inline Index NodeIdx(NodeID id) const {
    NDLL_ENFORCE_VALID_INDEX(id, (Index)id_to_node_map_.size());
    return id_to_node_map_[id].second;
  }

  /**
   * @brief Returns the TensorMeta objects for the tensor
   * with the given name and its producer node.
   */
  inline TensorMeta TensorSourceMeta(const string &name) const {
    auto it = tensor_producers_.find(name);
    NDLL_ENFORCE(it != tensor_producers_.end(), "Tensor with name \"" +
        name + "\" has no known source.");
    return it->second;
  }

  /**
   * @brief Checks if given Tensor already exists in the graph
   */
  inline bool TensorExists(const string &name) {
    auto it = tensor_producers_.find(name);
    return it != tensor_producers_.end();
  }

  /**
   * @brief Returns the id of the op that produces the tensor with
   * the given name.
   */
  inline NodeID TensorSourceID(const string &name) {
    return TensorSourceMeta(name).node;
  }

  /**
   * @brief Returns the output idx of the input tensor in
   * its source.
   */
  inline Index TensorIdxInSource(const string &name) {
    return TensorSourceMeta(name).index;
  }

  /**
   * @brief Returns true if the tensor with the given name
   * has a backend type that matches the calling type.
   */
  template <typename Backend>
  bool TensorIsType(const string &name);

  /**
   * @brief Returns a vector of meta-data about the nodes that
   * consume the tensor with the input name.
   */
  inline vector<TensorMeta> TensorConsumerMeta(const string &name) const {
    auto it = tensor_consumers_.find(name);
    if (it == tensor_consumers_.end()) {
      // If we have no entries for this tensors consumers,
      // we just return an empty vector
      return vector<TensorMeta>{};
    }
    return it->second;
  }


  /**
   * @brief Returns the OpNode at idx from the id to node
   * map.
   */
  const OpNode& GetNodeForIdx(int idx) const {
    NDLLOpType type = id_to_node_map_[idx].first;
    Index index = id_to_node_map_[idx].second;
    switch (type) {
    case NDLL_CPU:
      return cpu_nodes_[index];
    case NDLL_GPU:
      return gpu_nodes_[index];
    case NDLL_MIXED:
      return mixed_nodes_[index];
    }
    string str_error = "No Node for index " + idx;
    NDLL_FAIL(str_error);
  }

  // Helper for GraphTraversal
  std::string GetGoodLabel(const std::string& name) {
    std::size_t pos = name.find("__");
    NDLL_ENFORCE(pos != std::string::npos && pos + 2 < name.length());
    return name.substr(pos + 2);
  }

  void GraphTraversal(const OpNode& current_node, std::ofstream& ofs) {
    if (current_node.children.empty()
        || visited_nodes_.find(current_node.id) != visited_nodes_.end()) {
      ofs << GetGoodLabel(current_node.instance_name) << "\n";
      return;
    }
    visited_nodes_.insert(current_node.id);
    for (auto node_id: current_node.children) {
        ofs << GetGoodLabel(current_node.instance_name);
        ofs << " -> ";
        OpNode& child_node = node(node_id);
        GraphTraversal(child_node, ofs);
    }
  }

  /**
   * @brief Save graph in DOT directed graph format
   * in filename.
   */
  void SaveToDotFile(const string filename) {
    std::ofstream ofs = std::ofstream(filename);
    ofs << "digraph graphname {\n";
    const OpNode& current_node = GetNodeForIdx(0);
    GraphTraversal(current_node, ofs);
    ofs << "}\n";
    visited_nodes_.clear();
  }

  DISABLE_COPY_MOVE_ASSIGN(OpGraph);

 private:
  vector<OpNode> cpu_nodes_;
  vector<OpNode> gpu_nodes_;
  vector<OpNode> mixed_nodes_;

  // Stores a mapping from NodeIDs to a pair where the first
  // element indicates what type of node it is,  and the second
  // is the index of the op within the specified vector.
  vector<std::pair<NDLLOpType, Index>> id_to_node_map_;

  std::map<string, TensorMeta> tensor_producers_;
  std::map<string, vector<TensorMeta>> tensor_consumers_;

  // For the graph traversal
  std::unordered_set<NodeID> visited_nodes_;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OP_GRAPH_H_
