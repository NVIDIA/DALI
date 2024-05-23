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

namespace dali {
namespace graph {

struct DataNode;

struct OpNode {
  OpNode(std::string instance_name, OpSpec spec)
  : instance_name(std::move(instance_name)), spec(std::move(spec)) {}

  /** A visit marker for various graph processing algorithms */
  mutable bool visited = false;

  /** A unique name of an operator
   *
   * The string is `const` because it's view is used as a key.
   */
  const std::string instance_name;
  /** The specification of an operato */
  OpSpec spec;
  /** The "device" - cpu, gpu or mixed */
  OpType op_type;

  /** This node must not be pruned */
  bool keep = false;

  /** This list contains both positional and argument inputs */
  SmallVector<DataNode *, 8> inputs;
  /** This list contains the outputs of the operator (including unused ones) */
  SmallVector<DataNode *, 8> outputs;
};

struct DataEdge {
  /** The relevant operator (produced or consumer), depending on context. */
  OpNode *op = nullptr;
  /** The index of the producer's output or consumer's input, dependin on context. */
  int idx = 0;
};

struct DataNode {
  DataNode(std::string name, StorageDevice device) : name(std::move(name)), device(device) {}

  /** A visit marker for various graph processing algorithms */
  mutable bool visited = false;

  /** The name of the data node - typically operator name, output index and device
   *
   * The string is `const` because it's view is used as a key.
   */
  const std::string name;
  /** The storage device - CPU or GPU */
  StorageDevice device;

  /** The unique source of the data node*/
  DataEdge producer;
  /** Consumers of the data node */
  SmallVector<DataEdge, 4> consumers;

  /** True if the DataNode is a pipeline output; it may have other consumers, too. */
  bool pipeline_output = false;
};

class OpGraphBuilder;

/** A graph defining a pipeline.
 *
 * This graph represents the operators and connections between them.
 * It is lowered to executor-specific graph before being used for actually running something.
 */
class DLL_PUBLIC OpGraph {
 public:
  class DLL_PUBLIC Builder;
  friend class Builder;

  using OpNodeList = std::list<OpNode>;
  using DataNodeList = std::list<DataNode>;

  const OpNodeList &OpNodes() const {
    return op_nodes_;
  }

  const DataNodeList &DataNodes() const {
    return data_nodes_;
  }

  OpNode *GetOp(std::string_view instance_name) {
    auto it = name2op_.find(instance_name);
    if (it != name2op_.end())
      return &*it->second;
    else
      return nullptr;
  }

  DataNode *GetData(std::string_view data_node_name) {
    auto it = name2data_.find(data_node_name);
    if (it != name2data_.end())
      return &*it->second;
    else
      return nullptr;
  }

  void Prune();

  void Sort();

  OpNode &AddOp(std::string instance_name, OpSpec spec);

  DataNode &AddData(std::string name, StorageDevice device);

  bool EraseOp(std::string_view name);

  bool EraseData(std::string_view name);

  void MarkAsOutput(std::string_view name);

 private:
  bool has_active_builder_ = false;
  OpNodeList op_nodes_;
  DataNodeList data_nodes_;
  // The maps are keyed with `string_view` to avoid creation of temporary strings for lookup.
  // The string_view must refer to a live string, which, in this case the name of the node,
  // stored in the list.
  std::unordered_map<std::string_view, OpNodeList::iterator> name2op_;
  std::unordered_map<std::string_view, DataNodeList::iterator> name2data_;
  std::vector<DataNodeList::iterator> outputs_;
};

class DLL_PUBLIC OpGraph::Builder {
 public:
  void Add(std::string instance_name, OpSpec spec);

  /** Marks a data node with the given name as a pipeline output.
   *
   * @param output_name The output name, suffixed with the storage device, e.g. MyOp[0]_gpu.
   */
  void AddOutput(std::string output_name);

  void Build();
  OpGraph GetGraph() &&;
 private:
  bool built_ = false;
  std::vector<std::string> output_names_;
  OpGraph graph_;
};

}  // namespace graph
}  // namespace dali

#endif  // DALI_PIPELINE_GRAPH_OP_GRAPH2_H_
