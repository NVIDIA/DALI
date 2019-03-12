// Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_GRAPH_OP_GRAPH_H_
#define DALI_PIPELINE_GRAPH_OP_GRAPH_H_

#include <map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <set>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

using OpNodeId = int64_t;
using OpPartitionId = int64_t;
using TensorNodeId = int64_t;
using TensorPartitionId = int64_t;
// using producer_edge_t = std::pair<OpNodeId, Index>;
// using consumer_edge_t = std::pair<OpNodeId, Index>;


// What device is this tensor stored on
enum class StorageDevice {
  CPU = 0,
  GPU = 1,
  COUNT = 2,
};

template <StorageDevice>
struct storage_backend_type;

template <>
struct storage_backend_type<StorageDevice::CPU> {
  using type = CPUBackend;
};

template <>
struct storage_backend_type<StorageDevice::GPU> {
  using type = GPUBackend;
};

template <StorageDevice device>
using storage_backend_t = typename storage_backend_type<device>::type;

struct OpNode {
  inline OpNode() {}
  virtual ~OpNode() = default;
  OpNode& operator=(const OpNode&) = delete;

  OpNode(OpNode &&) = default;
  OpNode& operator=(OpNode &&) = default;

  inline OperatorBase &InstantiateOperator() {
    if (!op) op = dali::InstantiateOperator(spec);
    return *op;
  }

  std::unique_ptr<OperatorBase> op;
  OpNodeId id;
  OpSpec spec;
  std::set<OpNodeId> parents, children;

  // parent and children tensors indexed by our inputs and outputs
  std::vector<TensorNodeId> parent_tensors, children_tensors;

  std::string instance_name;
  OpType op_type;
  OpPartitionId partition_index;
};

// Stores meta-data about a tensor and how it
// is used by a producer/consumer node.
struct TensorMeta {
  OpNodeId node;
  Index index;
  StorageDevice storage_device;
  bool is_support;
};

using producer_edge_t = TensorMeta;
using consumer_edge_t = TensorMeta;

// Second type of graph nodes.
struct TensorNode {
  TensorNodeId id;
  std::string name;  // TODO(klecki): not happy about all the strings
  producer_edge_t producer;
  // order of consumers is arbitrary
  std::vector<consumer_edge_t> consumers;
};



/**
 * @brief Stores all meta-data about a graph of operations to be run
 * keeps track of useful meta-data about consumers/producers of
 * different intermediates.
 *
 * Operators in the graph have a global OpNodeId that is assigned in
 * the order ops are added to the graph. Operators also have an
 * index within the set of ops of its type (cpu, mixed, gpu).
 * This enables us to iterate over select portions of the graph, or
 * the entire graph.
 */
class DLL_PUBLIC OpGraph {
 public:
  DLL_PUBLIC inline OpGraph() {
    op_partitions_.resize(static_cast<int>(OpType::COUNT));
  }
  DLL_PUBLIC inline ~OpGraph() = default;

  /**
   * @brief Adds an op with the input specification to the graph.
   */
  DLL_PUBLIC void AddOp(const OpSpec &spec, const std::string& name);

  /**
   * @brief Removes the node with the specified OpNodeId from
   * the graph. Fails if the removal would produce an invalid
   * graph.
   */
  DLL_PUBLIC void RemoveOp(OpNodeId id);

  /**
   * @brief Returns the total number of ops in the graph.
   */
  DLL_PUBLIC inline Index NumOp() const {
    return op_nodes_.size();
  }

  /**
   * @brief Returns the total number of tensors in the graph.
   */
  DLL_PUBLIC inline Index NumTensor() const {
    return tensor_nodes_.size();
  }

  /**
   * @brief Returns the number of `op_type` ops in the graph.
   */
  DLL_PUBLIC inline Index NumOp(OpType op_type) const {
    return op_partitions_[static_cast<int>(op_type)].size();
  }

  /**
   * @brief Returns the unique NodeId for partition_id among nodes of op_type
   */
  DLL_PUBLIC inline OpNodeId NodeId(OpType op_type, OpPartitionId partition_id) const {
    DALI_ENFORCE_VALID_INDEX(partition_id, NumOp(op_type));
    return op_partitions_[static_cast<int>(op_type)][partition_id];
  }

  DLL_PUBLIC inline OpNode& Node(OpType op_type, OpPartitionId partition_id) {
    auto node_id = NodeId(op_type, partition_id);
    return op_nodes_[node_id];
  }

  DLL_PUBLIC inline const OpNode& Node(OpType op_type, OpPartitionId partition_id) const {
    auto node_id = NodeId(op_type, partition_id);
    return op_nodes_[node_id];
  }

  /**
   * @brief Returns the graph node with the given name.
   * This function is much slower than the version taking
   * index as argument so should not be used in performance
   * critical section of the code.
   */
  DLL_PUBLIC OpNode& Node(const std::string& name);

  /**
   * @brief Returns the graph node with the given index in the graph.
   */
  DLL_PUBLIC OpNode& Node(OpNodeId id) {
    DALI_ENFORCE_VALID_INDEX(id, op_nodes_.size());
    return op_nodes_[id];
  }

  /**
   * @brief Returns the graph node with the given index in the graph.
   */
  DLL_PUBLIC const OpNode& Node(OpNodeId id) const {
    DALI_ENFORCE_VALID_INDEX(id, op_nodes_.size());
    return op_nodes_[id];
  }

  DLL_PUBLIC TensorNode& Tensor(TensorNodeId id) {
    DALI_ENFORCE_VALID_INDEX(id, tensor_nodes_.size());
    return tensor_nodes_[id];
  }

  DLL_PUBLIC const TensorNode& Tensor(TensorNodeId id) const {
    DALI_ENFORCE_VALID_INDEX(id, tensor_nodes_.size());
    return tensor_nodes_[id];
  }

  DLL_PUBLIC const TensorNodeId TensorId(const std::string& name) const {
    auto it = tensor_name_to_id_.find(name);
    DALI_ENFORCE(it != tensor_name_to_id_.end(),
                 "Tensor with name " + name + " does not exist in graph.");
    return it->second;
  }

  /**
   * @brief Returns the Tensor node with the given name.
   */
  DLL_PUBLIC const TensorNode& Tensor(const std::string& name) const {
    return tensor_nodes_[TensorId(name)];
  }

  DLL_PUBLIC std::vector<std::vector<TensorNodeId>> PartitionTensorByOpType() const;

  /**
   * @brief Returns the type (cpu, gpu, mixed) of the node
   * at the given index.
   */
  DLL_PUBLIC inline OpType NodeType(OpNodeId id) const {
    DALI_ENFORCE_VALID_INDEX(id, (Index)op_nodes_.size());
    return op_nodes_[id].op_type;
  }

  /**
   * @brief Returns the index of the node with the specified id
   * among nodes of its type.
   */
  DLL_PUBLIC inline Index NodeIdx(OpNodeId id) const {
    DALI_ENFORCE_VALID_INDEX(id, (Index)op_nodes_.size());
    return op_nodes_[id].partition_index;
  }

  /**
   * @brief Returns the TensorMeta objects for the tensor
   * with the given name and its producer node.
   */
  DLL_PUBLIC inline TensorMeta TensorSourceMeta(const string &name) const {
    auto it = tensor_name_to_id_.find(name);
    DALI_ENFORCE(it != tensor_name_to_id_.end(), "Tensor with name \"" +
        name + "\" has no known source.");
    return tensor_nodes_[it->second].producer;
  }

  /**
   * @brief Checks if given Tensor already exists in the graph
   */
  DLL_PUBLIC inline bool TensorExists(const string &name) {
    auto it = tensor_name_to_id_.find(name);
    return it != tensor_name_to_id_.end();
  }

  /**
   * @brief Returns the id of the op that produces the tensor with
   * the given name.
   */
  DLL_PUBLIC inline OpNodeId TensorSourceID(const string &name) {
    return TensorSourceMeta(name).node;
  }

  /**
   * @brief Returns the output idx of the input tensor in
   * its source.
   */
  DLL_PUBLIC inline Index TensorIdxInSource(const string &name) {
    return TensorSourceMeta(name).index;
  }

  /**
   * @brief Returns true if the tensor with the given name
   * has a backend type that matches the calling type.
   */
  template <typename Backend>
  DLL_PUBLIC bool TensorIsType(const string &name);

  /**
   * @brief Returns a vector of meta-data about the nodes that
   * consume the tensor with the input name.
   */
  DLL_PUBLIC inline vector<TensorMeta> TensorConsumerMeta(const string &name) const {
    auto it = tensor_name_to_id_.find(name);
    if (it == tensor_name_to_id_.end()) {
      // If we have no entries for this tensors consumers,
      // we just return an empty vector
      return vector<TensorMeta>{};
    }
    return tensor_nodes_[it->second].consumers;
  }

  /**
   * @brief Helper function for saving graph to DOT file
   */
  DLL_PUBLIC void GenerateDOTFromGraph(std::ofstream& ofs, bool show_tensors, bool show_ids,
                                       bool use_colors);

  /**
   * @brief Instantiates the operators based on OpSpecs in nodes
   */
  DLL_PUBLIC void InstantiateOperators();

  /**
   * @brief Save graph in DOT directed graph format
   * in filename.
   */
  DLL_PUBLIC void SaveToDotFile(const string filename, bool show_tensors = false,
                                bool show_ids = false, bool use_colors = false) {
    std::ofstream ofs(filename);
    ofs << "digraph graphname {\n";
    GenerateDOTFromGraph(ofs, show_tensors, show_ids, use_colors);
    ofs << "}\n";
  }

  DLL_PUBLIC std::vector<TensorNodeId> GetOutputs(const std::vector<string>& output_names) const;
  DLL_PUBLIC std::vector<TensorNodeId> GetStageOutputs(OpType stage) const;

  DISABLE_COPY_MOVE_ASSIGN(OpGraph);

 private:
  // Should be called only once for each tensor
  void GenerateDOTFromGraph(const TensorNode& current_node, std::ofstream& ofs, bool show_tensors,
                            bool show_ids);

  /**
   * @brief Recalculate OpNodes partitioning
   *
   * Clears the partition vectors and readds all the nodes to proper partitions again,
   * storing new indexes.
   */
  void RepartitionOps();

  /**
   * @brief Adds new OpNode of `op_type` to op_nodes_ unified vector and to proper partition
   * storing its spec and instance_name. OpNode are given consecutive ids.
   *
   * @return Reference to the newly added OpNode.
   */
  OpNode& PlaceNewOp(OpType op_type, OpSpec op_spec, std::string instance_name);

  /**
   * @brief Creates new tensor node with conscutive id.
   *
   * @return Reference to the newly added tensor node.
   */
  TensorNode& PlaceNewTensor();


  std::vector<OpNode> op_nodes_;
  std::vector<TensorNode> tensor_nodes_;
  std::vector<std::vector<OpNodeId>> op_partitions_;

  /**
   * @brief  Swap ids of two TensorNodes, and update all occurences in graph to not break
   * any edges.
   *
   * @param left_id
   * @param right_id
   */
  void SwapTensorNodes(TensorNodeId left_id, TensorNodeId right_id);

  /**
   * @brief Remove consumerless TensorNode from graph.
   *
   * Order of other TensorNodes is preserved, with dense indexing.
   * @param id
   */
  void RemoveTensorNode(TensorNodeId id);

  /**
   * @brief Swap ids of two OpNodes, and update all occurences in graph to not break
   * any edges.
   *
   * @param left_id
   * @param right_id
   */
  void SwapOpNodes(OpNodeId left_id, OpNodeId right_id);

  /**
   * @brief Remove OpNode from graph
   *
   * Removed OpNode should not have any children, order of other OpNodes is preserved
   * with dense indexing
   * @param id
   */
  void RemoveOpNode(OpNodeId id);

  std::map<std::string, TensorNodeId> tensor_name_to_id_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_GRAPH_OP_GRAPH_H_
