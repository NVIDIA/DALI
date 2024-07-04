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

#include <gtest/gtest.h>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include "dali/pipeline/executor/executor2/stream_assignment.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/operator_factory.h"

namespace dali {

template <typename Backend>
class StreamAssignmentDummyOp : public Operator<Backend> {
 public:
  using Operator<Backend>::Operator;
  USE_OPERATOR_MEMBERS();

  void RunImpl(Workspace &ws) override {}
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return false;
  }
};

DALI_SCHEMA(StreamAssignmentDummyOp)
  .NumInput(0, 999)
  .NumOutput(0)
  .AdditionalOutputsFn([](const OpSpec &spec) {
    return spec.NumOutput();
  });

DALI_REGISTER_OPERATOR(StreamAssignmentDummyOp, StreamAssignmentDummyOp<CPUBackend>, CPU);

DALI_REGISTER_OPERATOR(StreamAssignmentDummyOp, StreamAssignmentDummyOp<GPUBackend>, GPU);

namespace exec2 {

namespace {

OpSpec SpecDev(const std::string &device) {
  return OpSpec("StreamAssignmentDummyOp")
    .AddArg("device", device)
    .AddArg("num_threads", 1)
    .AddArg("max_batch_size", 1);
}

OpSpec SpecGPU() {
  return SpecDev("gpu");
}

OpSpec SpecCPU() {
  return SpecDev("cpu");
}

auto MakeNodeMap(const ExecGraph &graph) {
  std::map<std::string_view, const ExecNode *, std::less<>> map;
  for (auto &n : graph.nodes)
    if (n.def) {
      map[n.def->instance_name] = &n;
    }
  return map;
}

}  // namespace

TEST(StreamAssignmentTest, PerOperator) {
  ExecGraph eg;
  /*
         --c--             g
        /     \         /   \
  a -- b ----- d ----- f ---- h ---> out
   \   (cpu)       /       /
    --------------e        /
                        /
  i ----------------- j(cpu)

  k ----------------------------> out

  */
  graph::OpGraph::Builder b;
  b.Add("a",
        SpecGPU()
        .AddOutput("a->b", "gpu")
        .AddOutput("a->e", "gpu"));
  b.Add("i",
        SpecGPU()
        .AddOutput("i->j", "gpu"));
  b.Add("j",
        SpecCPU()
        .AddInput("i->j", "gpu")
        .AddOutput("j->h", "cpu"));
  b.Add("b",
        SpecCPU()
        .AddInput("a->b", "gpu")
        .AddOutput("b->c", "cpu")
        .AddOutput("b->d", "cpu"));
  b.Add("c",
        SpecGPU()
        .AddInput("b->c", "cpu")
        .AddOutput("c->d", "gpu"));
  b.Add("d",
        SpecGPU()
        .AddInput("b->d", "cpu")
        .AddInput("c->d", "gpu")
        .AddOutput("d->f", "gpu"));
  b.Add("e",
        SpecGPU()
        .AddInput("a->e", "gpu")
        .AddOutput("e->f", "gpu"));
  b.Add("f",
        SpecGPU()
        .AddInput("d->f", "gpu")
        .AddInput("e->f", "gpu")
        .AddOutput("f->g", "gpu")
        .AddOutput("f->h", "gpu"));
  b.Add("g",
        SpecGPU()
        .AddInput("f->g", "gpu")
        .AddOutput("g->h", "gpu"));
  b.Add("h",
        SpecGPU()
        .AddInput("f->h", "gpu")
        .AddInput("g->h", "gpu")
        .AddInput("j->h", "cpu")
        .AddOutput("h->o", "gpu"));
  b.Add("k",
        SpecGPU()
        .AddOutput("k->o", "gpu"));  // directly to output
  b.AddOutput("h->o_gpu");
  b.AddOutput("k->o_gpu");
  auto g = std::move(b).GetGraph(true);
  eg.Lower(g);

  StreamAssignment<StreamPolicy::PerOperator> assignment(eg);
  auto map = MakeNodeMap(eg);
  EXPECT_EQ(assignment[map["a"]], 0);
  EXPECT_EQ(assignment[map["b"]], std::nullopt);
  EXPECT_EQ(assignment[map["c"]], 0);
  EXPECT_EQ(assignment[map["d"]], 0);
  EXPECT_EQ(assignment[map["e"]], 3);
  EXPECT_EQ(assignment[map["f"]], 0);
  EXPECT_EQ(assignment[map["g"]], 0);
  EXPECT_EQ(assignment[map["h"]], 0);
  EXPECT_EQ(assignment[map["i"]], 1);
  EXPECT_EQ(assignment[map["j"]], std::nullopt);
  EXPECT_EQ(assignment[map["k"]], 2);
}

}  // namespace exec2
}  // namespace dali
