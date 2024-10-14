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
#include <utility>
#include "dali/core/error_handling.h"
#include "dali/pipeline/operator/name_utils.h"

namespace dali {
namespace graph {

//////////////////////////////////////////////////////////////////////////////
// OpGraph

OpNode &OpGraph::AddOp(std::string instance_name, OpSpec spec) {
  OpNodeList tmp;
  OpType type = OpType::CPU;
  std::string device;
  if (spec.TryGetArgument(device, "device"))
    type = ParseOpType(device);
  bool preserve = spec.GetArgument<bool>("preserve") || spec.GetSchemaOrDefault().IsNoPrune();
  auto &op_node = tmp.emplace_back(std::move(instance_name), type, std::move(spec));
  if (!name2op_.emplace(op_node.instance_name, &op_node).second) {
    throw std::invalid_argument(
        make_string("Duplicate operator instance name: \"", op_node.instance_name, "\""));
  }
  op_node.iter = tmp.begin();

  op_node.keep = preserve;
  op_nodes_.splice(op_nodes_.end(), tmp);
  return op_node;
}

DataNode &OpGraph::AddData(std::string name, StorageDevice device) {
  DataNodeList tmp;
  auto &data_node = tmp.emplace_back(std::move(name), device);
  if (!name2data_.emplace(data_node.name, &data_node).second) {
    throw std::invalid_argument(
        make_string("Duplicate data node name: \"", data_node.name, "\""));
  }
  data_node.iter = tmp.begin();
  data_nodes_.splice(data_nodes_.end(), tmp);
  return data_node;
}

void OpGraph::RemoveDataNodeReferences(OpNode &op) {
  for (auto *data : op.inputs) {
    auto it = std::remove_if(data->consumers.begin(), data->consumers.end(), [&](DataEdge e) {
      return e.op == &op;
    });
    // the removed op must have been a consumer
    assert(it != data->consumers.end());
    data->consumers.erase(it, data->consumers.end());
  }
  for (auto *data : op.outputs) {
    // the removed op must have been the producer
    assert(data->producer.op == &op);
    data->producer = {};
  }
}


bool OpGraph::EraseOp(std::string_view name) {
  auto it = name2op_.find(name);
  if (it == name2op_.end())
    return false;
  auto &op = *it->second;

  RemoveDataNodeReferences(op);

  op_nodes_.erase(it->second->iter);
  name2op_.erase(it);
  return true;
}


bool OpGraph::EraseData(std::string_view name) {
  auto it = name2data_.find(name);
  if (it == name2data_.end())
    return false;

  auto &node = *it->second;
  if (node.producer.op) {
    assert(node.producer.op->outputs.size() >= static_cast<size_t>(node.producer.idx));
    assert(node.producer.op->outputs[node.producer.idx] == &node);
    node.producer.op->outputs[node.producer.idx] = nullptr;
  }

  for (auto &edge : node.consumers) {
    if (edge.op == nullptr)
      continue;
    assert(edge.op->inputs.size() >= static_cast<size_t>(edge.idx));
    assert(edge.op->inputs[edge.idx] == &node);
    edge.op->inputs[edge.idx] = nullptr;
  }

  data_nodes_.erase(it->second->iter);
  name2data_.erase(it);
  return true;
}

int OpGraph::AddOutput(std::string_view name) {
  auto it = name2data_.find(name);
  if (it == name2data_.end())
    throw std::invalid_argument(make_string(
      "The name \"", name, "\" is not a name of a known DataNode."));
  it->second->pipeline_output = true;
  outputs_.push_back(it->second->name);
  return outputs_.size() - 1;
}

/** Implements topological sorting via depth-first search */
class OpGraph::SortHelper {
 public:
  explicit SortHelper(OpGraph &graph) : graph_(graph) {
    sorted_ops_.reserve(graph_.op_nodes_.size());
    sorted_data_.reserve(graph_.data_nodes_.size());
  }

  void Run(bool prune) {
    ClearVisitMarkers(graph_.op_nodes_);
    ClearVisitMarkers(graph_.data_nodes_);

    // First, go over the outputs
    for (auto &data : graph_.data_nodes_) {
      if (data.pipeline_output)
        Traverse(&data);
    }

    // Then add operators marked as "keep".
    for (auto &op : graph_.op_nodes_) {
      if (op.keep)
        Traverse(&op);
    }

    if (!prune) {
      // Not pruning? Add remaning operators and data nodes.
      // They will appear after all relevant nodes.
      for (auto &op : graph_.op_nodes_) {
        Traverse(&op);
      }

      for (auto &data : graph_.data_nodes_) {
        Traverse(&data);
      }
    }

    OpNodeList out_ops;
    for (auto *op : sorted_ops_)
      out_ops.splice(out_ops.end(), graph_.op_nodes_, op->iter);

    DataNodeList out_data;
    for (auto *data : sorted_data_)
      out_data.splice(out_data.end(), graph_.data_nodes_, data->iter);

    graph_.op_nodes_.swap(out_ops);
    graph_.data_nodes_.swap(out_data);

    // Now out_ops and out_data are the ops and data, respectively, that got removed.
    // We should remove them from the name2xx maps and remove corresponding consumer links.
    // NOTE: no producer of otherwise valid node should be removed, so we only need to adjust
    // consumers.

    for (auto &pruned_op : out_ops) {
      graph_.RemoveDataNodeReferences(pruned_op);
      graph_.name2op_.erase(pruned_op.instance_name);
    }
    for (auto &pruned_data : out_data) {
      graph_.name2data_.erase(pruned_data.name);
    }
  }

 private:
  OpGraph &graph_;

  std::vector<OpNode *> sorted_ops_;
  std::vector<DataNode *> sorted_data_;

  void Traverse(OpNode *op) {
    Visit visit(op);
    if (!visit)
      return;
    for (auto *inp : op->inputs) {
      Traverse(inp);
    }
    sorted_ops_.push_back(op);
    for (auto *out : op->outputs)
      if (Visit(out))
        sorted_data_.push_back(out);
  }

  void Traverse(DataNode *data) {
    if (!Visit(data))
      return;
    if (data->producer.op)
      Traverse(data->producer.op);
    sorted_data_.push_back(data);
  }
};

void OpGraph::Sort(bool prune) {
  SortHelper sort(*this);
  sort.Run(prune);
}

//////////////////////////////////////////////////////////////////////////////
// OpGraph::Builder

void OpGraph::Builder::AddOutput(std::string name) {
  output_names_.push_back(std::move(name));
}

void OpGraph::Builder::Add(std::string instance_name, OpSpec new_spec) {
  auto &op_node = graph_.AddOp(std::move(instance_name), std::move(new_spec));

  const OpSpec &spec = op_node.spec;

  for (int i = 0; i < spec.NumInput(); i++) {
    std::string name = spec.Input(i);
    auto it = graph_.name2data_.find(name);
    DataNode *node;
    if (it != graph_.name2data_.end()) {
      node = it->second;
    } else {
      auto dev = ParseStorageDevice(spec.InputDevice(i));
      node = &graph_.AddData(name, dev);
    }
    node->consumers.push_back({ &op_node, i });
    op_node.inputs.push_back(node);
  }

  for (int o = 0; o < spec.NumOutput(); o++) {
    std::string name = spec.Output(o);
    auto it = graph_.name2data_.find(name);
    DataNode *node;
    if (it != graph_.name2data_.end()) {
      node = it->second;
      if (node->producer.op != nullptr) {
        throw std::invalid_argument(make_string(
          "The data node \"", name, "\" has more than one producer:"
          "\n 1: ", node->producer.op->instance_name, ", output ", node->producer.idx,
          "\n 2: ", op_node.instance_name, ", output ", o));
      }
    } else {
      auto dev = ParseStorageDevice(spec.OutputDevice(o));
      node = &graph_.AddData(std::move(name), dev);
    }
    node->producer = { &op_node, o };
    op_node.outputs.push_back(node);
  }
}

void OpGraph::Builder::Build(bool prune) {
  if (!built_) {
    for (auto &out : output_names_)
      graph_.AddOutput(out);
    graph_.Sort(prune);
    built_ = true;
    pruned_ = prune;
  }
  if (prune && !pruned_) {
    graph_.Sort(true);
    pruned_ = true;
  }
}

OpGraph OpGraph::Builder::GetGraph(bool prune) && {
  Build(prune);
  return std::move(graph_);
}

}  // namespace graph
}  // namespace dali
