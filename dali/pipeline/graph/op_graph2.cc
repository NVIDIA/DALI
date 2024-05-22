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

namespace dali {

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

void OpGraph2::Builder::Add(std::string instance_name, const OpSpec &spec) {
  if (graph_.name2op_.count(instance_name) > 0) {
    throw std::invalid_argument(
        make_string("Duplicate operator instance name: \"", instance_name, "\""));
  }

  auto &node = graph_.op_nodes_.emplace_back(std::move(instance_name), spec);
  graph_.name2op_.emplace(node.instance_name, std::prev(graph_.op_nodes_.end()));
  graph_.
}

}  // namespace dali
