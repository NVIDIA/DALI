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

#ifndef DALI_PIPELINE_GRAPH_OP_GRAPH2_H_
#define DALI_PIPELINE_GRAPH_OP_GRAPH2_H_

#include <cassert>
#include <list>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <utility>
#include "dali/core/common.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/graph/graph_util.h"

namespace dali {
namespace graph {

class OpGraph;

struct OpNode;
struct DataNode;

using OpNodeList = std::list<OpNode>;
using DataNodeList = std::list<DataNode>;

class OpGraph;

struct OpNode {
 private:
  /** The iterator that points to this object in the enclosing list. */
  OpNodeList::iterator iter{};

 public:
  friend class OpGraph;

  OpNode(std::string instance_name, OpType type, OpSpec spec)
  : instance_name(std::move(instance_name)), spec(std::move(spec)), op_type(type) {}

  /** A visit marker for various graph processing algorithms. */
  mutable bool visited = false;
  /** A visit marker for cycle detection. */
  mutable bool visit_pending = false;

  /** A unique name of an operator
   *
   * The string is `const` because it's view is used as a key.
   */
  const std::string instance_name;
  /** The specification of an operator. */
  OpSpec spec;
  /** The "device" - cpu, gpu or mixed. */
  OpType op_type = OpType::CPU;

  /** This node must not be pruned. */
  bool keep = false;

  /** This list contains both positional and argument inputs. */
  SmallVector<DataNode *, 8> inputs;
  /** This list contains the outputs of the operator (including unused ones). */
  SmallVector<DataNode *, 4> outputs;
};

struct DataEdge {
  /** The relevant operator (producer or consumer), depending on context. */
  OpNode *op = nullptr;
  /** The index of the producer's output or consumer's input, depending on context. */
  int idx = 0;
};

struct DataNode {
 private:
  /** The iterator that points to this object in the enclosing list. */
  DataNodeList::iterator iter{};

 public:
  friend class OpGraph;

  DataNode(std::string name, StorageDevice device) : name(std::move(name)), device(device) {}

  /** A visit marker for various graph processing algorithms. */
  mutable bool visited = false;
  /** A visit marker for cycle detection. */
  mutable bool visit_pending = false;

  /** The name of the data node - typically operator name, output index and device.
   *
   * The string is `const` because it's view is used as a key.
   */
  const std::string name;
  /** The storage device - CPU or GPU. */
  StorageDevice device;

  /** The unique source of the data node. */
  DataEdge producer;
  /** Consumers of the data node */
  SmallVector<DataEdge, 4> consumers;

  /** True if the DataNode is a pipeline output; it may have other consumers, too. */
  bool pipeline_output = false;
};

/** A graph defining a pipeline.
 *
 * This graph represents the operators and connections between them.
 * It is lowered to executor-specific graph before being used for actually running something.
 */
class DLL_PUBLIC OpGraph {
 public:
  class DLL_PUBLIC Builder;
  friend class Builder;

  const OpNodeList &OpNodes() const {
    return op_nodes_;
  }

  const DataNodeList &DataNodes() const {
    return data_nodes_;
  }

  OpNodeList &OpNodes() {
    return op_nodes_;
  }

  DataNodeList &DataNodes() {
    return data_nodes_;
  }

  /** Returns an OpNode with a matching instance name or nullptr. */
  OpNode *GetOp(std::string_view instance_name) {
    return GetOpImpl(instance_name);
  }

  /** Returns an OpNode with a matching instance name or nullptr. */
  const OpNode *GetOp(std::string_view instance_name) const {
    return GetOpImpl(instance_name);
  }

  /** Returns a DataNode with a matching name or nullptr. */
  DataNode *GetData(std::string_view data_node_name) {
    return GetDataImpl(data_node_name);
  }

  /** Returns a DataNode with a matching name or nullptr. */
  const DataNode *GetData(std::string_view data_node_name) const {
    return GetDataImpl(data_node_name);
  }

  /** Sorts the graph topologically and removes entries that do not contribute to essential nodes.
   *
   * After this function succeeds, the graph is topologically sorted from input to output.
   * The graph can be also pruned, keeping only the nodes that contribute to the outputs or one
   * of the operators with `keep` flag set.
   */
  void Sort(bool prune);

  /** Adds an operator node.
   *
   * This function doesn't create any DataNodes or validate anything. Use `OpGraph::Buidler`
   * to construct the graph from `OpSpecs`.
   */
  OpNode &AddOp(std::string instance_name, OpSpec spec);

  /** Adds a DataNode with given name and storage device.
   *
   * @name The full name of the data node. The storage device is not added to the name - it must
   *       be already there if desired.
   */
  DataNode &AddData(std::string name, StorageDevice device);

  /** Erases an operator node.
   *
   * Removes the operator node map entry and the corresponding list entry.
   * Consumer/producer pointers in referring data nodes are removed or cleared.
   */
  bool EraseOp(std::string_view name);

  /** Erases a data node.
   *
   * Removes the data node map entry and the corresponding list entry.
   * Corresponding entries in producer and consumers are replaced with null pointers.
   */
  bool EraseData(std::string_view name);

  /** Adds a pipeline output.
   *
   * @param name  The name of the data node to be bound to a new output.
   * @throws std::invalid_argument if the name does not correspond to a known DataNode.
   * @return The index of the newly added output.
   */
  int AddOutput(std::string_view name);

  span<const std::string_view> Outputs() const {
    return make_cspan(outputs_);
  }

 private:
  OpNode *GetOpImpl(std::string_view instance_name) const {
    auto it = name2op_.find(instance_name);
    if (it != name2op_.end())
      return &*it->second;
    else
      return nullptr;
  }

  DataNode *GetDataImpl(std::string_view data_node_name) const {
    auto it = name2data_.find(data_node_name);
    if (it != name2data_.end())
      return &*it->second;
    else
      return nullptr;
  }

  void RemoveDataNodeReferences(OpNode &op);

  OpNodeList op_nodes_;
  DataNodeList data_nodes_;
  // The maps are keyed with `string_view` to avoid creation of temporary strings for lookup.
  // The string_view must refer to a live string, which, in this case, is the name of the node,
  // stored in the list. std::list never invalidates pointers and the name is const, which
  // guarantees that the view will always remain correct.
  std::unordered_map<std::string_view, OpNode *> name2op_;
  std::unordered_map<std::string_view, DataNode *> name2data_;
  std::vector<std::string_view> outputs_;

  class SortHelper;
  friend class SortHelper;
};

/** A single-use class for constructing graphs. */
class DLL_PUBLIC OpGraph::Builder {
 public:
  /** Constructs an OpNode and all relevant DataNodes based on the operator's specification.
   *
   * This function creates an OpNode and goes over all inputs and outputs in the OpSpec,
   * updating or creating the relevant DataNodes.
   */
  void Add(std::string instance_name, OpSpec spec);

  /** Marks a data node with the given name as a pipeline output.
   *
   * @param output_name The output name, suffixed with the storage device, e.g. MyOp[0]_gpu.
   */
  void AddOutput(std::string output_name);

  void Build(bool prune);
  /**  Removes the newly created graph from the builder. */
  OpGraph GetGraph(bool prune = false) &&;
 private:
  bool built_ = false;
  bool pruned_ = false;
  std::vector<std::string> output_names_;
  OpGraph graph_;
};

}  // namespace graph
}  // namespace dali

#endif  // DALI_PIPELINE_GRAPH_OP_GRAPH2_H_
