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

#include "dali/pipeline/graph/cse.h"
#include "dali/pipeline/dali.pb.h"
#include <functional>
#include <map>
#include <string>

namespace dali {
namespace graph {

std::string OpSpecToString(const OpSpec &spec) {
  dali_proto::OpDef op;
  op.set_name(spec.SchemaName());

  for (int i = 0; i < spec.NumInput(); ++i) {
    dali_proto::InputOutput *in = op.add_input();
    in->set_name(spec.InputName(i));
    in->set_device(spec.InputDevice(i));
    if (spec.IsArgumentInput(i)) {
        in->set_arg_name(spec.ArgumentInputName(i));
    }
    in->set_is_argument_input(spec.IsArgumentInput(i));
  }

  for (int i = 0; i < spec.NumOutput(); ++i) {
    dali_proto::InputOutput *out = op.add_output();
    // clear output name!
    out->set_name(std::to_string(i));
    out->set_device(spec.OutputDevice(i));
    out->set_is_argument_input(false);
  }

  auto &schema = spec.GetSchemaOrDefault();
  for (auto& a : spec.Arguments()) {
    // filter out args that need to be dealt with on
    // loading a serialized pipeline
    auto name = a->get_name();

    // Some arguments should be skipped when comparing operators
    if (schema.GetArgument(name).ignore_cmp)
      continue;

    dali_proto::Argument *arg = op.add_args();
    DaliProtoPriv arg_wrap(arg);

    a->SerializeToProtobuf(&arg_wrap);
  }
  return op.SerializeAsString();
}

class CSE {
 public:
  explicit CSE(OpGraph &graph) : graph_(graph) {}

  void Run() {
  }

  void Run(OpNode *node) {
    for (int i = 0, ninp = node->inputs; i < ninp; ++i) {
    }
  }

  std::map<std::string, OpNode *> normalized_nodes_;
  std::map<OpNode *, OpNode *> renamed_;
  OpGraph &graph_;
  OpGraph::Builder builder_;
};

void EliminateCommonSubgraphs(OpGraph &graph) {
}

}  // namespace graph
}  // namespace dali
