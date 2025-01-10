// Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/graph/cse.h"
#include <functional>
#include <map>
#include <string>
#include <utility>
#include "dali/pipeline/dali.pb.h"

namespace dali {
namespace graph {

namespace {

/** Computes the CSE key by serializing the relevant subset of an OpSpec to protobuf */
std::string OpSpecCSEKey(const OpSpec &spec) {
  dali_proto::OpDef op;
  op.set_name(spec.SchemaName());

  for (int i = 0; i < spec.NumInput(); ++i) {
    dali_proto::InputOutput *in = op.add_input();
    in->set_name(spec.InputName(i));
    in->set_device(to_string(spec.InputDevice(i)));
    if (spec.IsArgumentInput(i)) {
        in->set_arg_name(spec.ArgumentInputName(i));
    }
    in->set_is_argument_input(spec.IsArgumentInput(i));
  }

  for (int i = 0; i < spec.NumOutput(); ++i) {
    dali_proto::InputOutput *out = op.add_output();
    // Use a placeholder instead of the real name
    out->set_name(std::to_string(i));
    out->set_device(to_string(spec.OutputDevice(i)));
  }

  auto &schema = spec.GetSchemaOrDefault();
  std::map<std::string_view, Argument *, std::less<>> sorted_args;
  for (auto &a : spec.Arguments()) {
    // Some arguments should be skipped when comparing operators
    auto arg_name = a->get_name();
    if (schema.HasArgument(arg_name))
      if (schema.GetArgument(arg_name).ignore_cmp)
        continue;

    sorted_args.emplace(arg_name, a.get());
  }

  for (auto [name, a] : sorted_args) {
    dali_proto::Argument *arg = op.add_args();
    DaliProtoPriv arg_wrap(arg);
    a->SerializeToProtobuf(&arg_wrap);
  }

  return op.SerializeAsString();
}

/** The context for Common Subgraph Elimination */
class CSE {
 public:
  void Run(OpGraph &graph) {
    for (auto &node : graph.OpNodes())
      Run(&node);
    for (auto output_name : graph.Outputs()) {
      auto it = renamed_full_.find(output_name);

      if (it != renamed_full_.end())
        builder_.AddOutput(it->second);
      else
        builder_.AddOutput(std::string(output_name));
    }
    graph = {};
    graph = std::move(builder_).GetGraph(true);
  }

  bool IsFoldable(const OpSpec &spec) {
    return !spec.GetArgument<bool>("preserve") &&
           !spec.GetArgument<bool>("preserve_name") &&
           !spec.GetSchemaOrDefault().IsNoPrune();
  }

  void Run(OpNode *node) {
    OpSpec new_spec = node->spec;
    for (int i = 0; i < new_spec.NumInput(); i++) {
      auto it = renamed_.find(new_spec.InputName(i));
      if (it != renamed_.end())
        new_spec.RenameInput(i, it->second);
    }
    std::string key = OpSpecCSEKey(new_spec);
    OpNode *&norm = normalized_nodes_[key];
    bool foldable = IsFoldable(new_spec);

    if (!norm || !foldable)
      norm = node;

    if (norm != node) {
      for (int o = 0; o < node->spec.NumOutput(); o++) {
        renamed_.emplace(node->spec.OutputName(o), norm->spec.OutputName(o));
        renamed_full_.emplace(node->spec.Output(o), norm->spec.Output(o));
      }
    } else {
      builder_.Add(norm->instance_name, new_spec);
    }
  }

  std::map<std::string, OpNode *> normalized_nodes_;
  std::map<std::string, std::string, std::less<>> renamed_;
  std::map<std::string, std::string, std::less<>> renamed_full_;
  OpGraph::Builder builder_;
};

}  // namespace

void EliminateCommonSubgraphs(OpGraph &graph) {
  CSE cse;
  cse.Run(graph);
}

}  // namespace graph
}  // namespace dali
