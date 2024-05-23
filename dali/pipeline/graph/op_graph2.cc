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

namespace {

// TODO(klecki): Graph creation is not a place to check OpSpec?
void CheckOpConstraints(const OpSpec &spec) {
  const OpSchema &schema = SchemaRegistry::GetSchema(spec.SchemaName());

  const int additional_outputs = schema.CalculateAdditionalOutputs(spec);

  DALI_ENFORCE(schema.SupportsInPlace(spec) || !spec.GetArgument<bool>("inplace"),
               make_string("Operator `", GetOpDisplayName(spec, true),
                           "` does not support in-place execution."));
  DALI_ENFORCE(
      spec.NumRegularInput() <= schema.MaxNumInput(),
      make_string("Operator `", GetOpDisplayName(spec, true), "` supports a maximum of ",
                  schema.MaxNumInput(), " inputs, but was passed ", spec.NumRegularInput(), "."));
  DALI_ENFORCE(
      spec.NumRegularInput() >= schema.MinNumInput(),
      make_string("Operator `", GetOpDisplayName(spec, true), "` supports a minimum of ",
                  schema.MinNumInput(), " inputs, but was passed ", spec.NumRegularInput(), "."));
  DALI_ENFORCE(spec.NumOutput() == schema.CalculateOutputs(spec) + additional_outputs,
               make_string("Operator `", GetOpDisplayName(spec, true), "` supports ",
                           schema.CalculateOutputs(spec) + additional_outputs,
                           " outputs, but was passed ", spec.NumOutput(), "."));
}

OpType ParseOpType(const std::string &device) {
  if (device == "gpu") {
    return OpType::GPU;
  } else if (device == "cpu") {
    return OpType::CPU;
  } else if (device == "mixed") {
    return OpType::MIXED;
  }
  DALI_FAIL("Unsupported device type: " + device + ".");
}

StorageDevice ParseStorageDevice(const std::string &io_device) {
  if (io_device == "cpu") {
    return StorageDevice::CPU;
  }
  return StorageDevice::GPU;
}

}  // namespace

OpNode &OpGraph::AddOp(std::string instance_name, OpSpec spec) {
  OpNodeList tmp;
  auto &op_node = tmp.emplace_back(std::move(instance_name), std::move(spec));
  if (!name2op_.emplace(op_node.instance_name, tmp.begin()).second) {
    throw std::invalid_argument(
        make_string("Duplicate operator instance name: \"", op_node.instance_name, "\""));
  }
  op_nodes_.splice(op_nodes_.end(), tmp);
  return op_node;
}

DataNode &OpGraph::AddData(std::string name, StorageDevice device) {
  DataNodeList tmp;
  auto &data_node = tmp.emplace_back(std::move(name), device);
  if (!name2data_.emplace(data_node.name, tmp.begin()).second) {
    throw std::invalid_argument(
        make_string("Duplicate data node name: \"", data_node.name, "\""));
  }
  data_nodes_.splice(data_nodes_.end(), tmp);
  return data_node;
}


bool OpGraph::EraseOp(std::string_view name) {
  auto it = name2op_.find(name);
  if (it == name2op_.end())
    return false;
  auto &op = *it->second;

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

  op_nodes_.erase(it->second);
  name2op_.erase(it);
  return true;
}


bool OpGraph::EraseData(std::string_view name) {
  auto it = name2data_.find(name);
  if (it == name2data_.end())
    return false;
  data_nodes_.erase(it->second);
  name2data_.erase(it);
  return true;
}

void OpGraph::Builder::MarkAsOutput(std::string_view name) {
  auto it = name2data_.find(name);
  if (it == name2data_.end())
    throw std::invalid_argument(make_string(
      "The name \"", name, "\" is not a name of a known DataNode."));
  outputs_.push_back(it);
}

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
      node = &*it->second;
    } else {
      auto dev = ParseStorageDevice(spec.InputDevice(i));
      node = &graph_.AddData(name, dev);
    }
    node->consumers.push_back({ &op_node, i });
  }

  for (int o = 0; o < spec.NumOutput(); o++) {
    std::string name = spec.Output(o);
    auto it = graph_.name2data_.find(name);
    if (it != graph_.name2data_.end()) {
      if (it->second->producer.op != nullptr) {
        throw std::invalid_argument(make_string(
          "The data node \"", name, "\" has more than one producer:"
          "\n 1: ", it->second->producer.op->instance_name, ", output ", it->second->producer.idx,
          "\n 2: ", instance_name, ", output ", o));
      }
    } else {
      auto dev = ParseStorageDevice(spec.OutputDevice(o));
      auto &node = graph_.AddData(std::move(name), dev);
      node.producer = { &op_node, o };
    }
  }

}

void OpGraph::Builder::Build() {
  if (built_)
    return true;
  for (auto &out : output_names_)
    graph_.AddOutput(out);
  built_ = true;
}

OpGraph OpGraph::Builder::GetGraph() && {
  if (!built_)
    Build();
  return std::move(graph_);
}

}  // namespace graph
}  // namespace dali
