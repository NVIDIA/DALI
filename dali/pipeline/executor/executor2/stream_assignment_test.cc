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
DALI_REGISTER_OPERATOR(StreamAssignmentDummyOp, StreamAssignmentDummyOp<MixedBackend>, Mixed);
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

OpSpec SpecMixed() {
  return SpecDev("mixed");
}

auto MakeNodeMap(const ExecGraph &graph) {
  std::map<std::string_view, const ExecNode *, std::less<>> map;
  for (auto &n : graph.Nodes())
    if (!n.instance_name.empty()) {
      map[n.instance_name] = &n;
    }
  return map;
}

}  // namespace

TEST(Exec2Test, StreamAssignment_Single_OnlyCPU) {
  graph::OpGraph::Builder b;
  b.Add("a",
        SpecCPU()
        .AddOutput("a->out", "cpu"));
  b.AddOutput("a->out_cpu");
  auto g = std::move(b).GetGraph(true);
  ExecGraph eg;
  eg.Lower(g);

  StreamAssignment<StreamPolicy::Single> assignment(eg);
  auto map = MakeNodeMap(eg);
  EXPECT_EQ(assignment[map["a"]], std::nullopt);
}

TEST(Exec2Test, StreamAssignment_Single_CPUMixedGPU) {
  graph::OpGraph::Builder b;
  b.Add("a",
        SpecCPU()
        .AddOutput("a->b", "cpu"));
  b.Add("b",
        SpecMixed()
        .AddInput("a->b", "cpu")
        .AddOutput("b->c", "gpu"));
  b.Add("c",
        SpecGPU()
        .AddInput("b->c", "gpu")
        .AddOutput("c->out", "gpu"));
  b.AddOutput("c->out_gpu");
  auto g = std::move(b).GetGraph(true);
  ExecGraph eg;
  eg.Lower(g);

  StreamAssignment<StreamPolicy::Single> assignment(eg);
  auto map = MakeNodeMap(eg);
  EXPECT_EQ(assignment[map["a"]], std::nullopt);
  EXPECT_EQ(assignment[map["b"]], 0);
  EXPECT_EQ(assignment[map["c"]], 0);
}


TEST(Exec2Test, StreamAssignment_PerBackend_OnlyCPU) {
  graph::OpGraph::Builder b;
  b.Add("a",
        SpecCPU()
        .AddOutput("a->out", "cpu"));
  b.AddOutput("a->out_cpu");
  auto g = std::move(b).GetGraph(true);
  ExecGraph eg;
  eg.Lower(g);

  StreamAssignment<StreamPolicy::Single> assignment(eg);
  auto map = MakeNodeMap(eg);
  EXPECT_EQ(assignment[map["a"]], std::nullopt);
}


TEST(Exec2Test, StreamAssignment_PerBackend_CPUMixed) {
  graph::OpGraph::Builder b;
  b.Add("a",
        SpecCPU()
        .AddOutput("a->b", "cpu")
        .AddOutput("a->c", "cpu"));
  b.Add("b",
        SpecMixed()
        .AddInput("a->b", "cpu")
        .AddOutput("b->out", "gpu"));
  b.Add("c",
        SpecMixed()
        .AddInput("a->c", "cpu")
        .AddOutput("c->out", "gpu"));
  b.AddOutput("b->out_gpu");
  b.AddOutput("c->out_gpu");
  auto g = std::move(b).GetGraph(true);
  ExecGraph eg;
  eg.Lower(g);

  StreamAssignment<StreamPolicy::PerBackend> assignment(eg);
  auto map = MakeNodeMap(eg);
  EXPECT_EQ(assignment[map["a"]], std::nullopt);
  EXPECT_EQ(assignment[map["b"]], 0);
  EXPECT_EQ(assignment[map["c"]], 0);
}

TEST(Exec2Test, StreamAssignment_PerBackend_CPUMixedGPU) {
  graph::OpGraph::Builder b;
  b.Add("a",
        SpecCPU()
        .AddOutput("a->b", "cpu")
        .AddOutput("a->c", "cpu"));
  b.Add("b",
        SpecGPU()
        .AddInput("a->b", "cpu")
        .AddOutput("b->out", "gpu"));
  b.Add("c",
        SpecMixed()
        .AddInput("a->c", "cpu")
        .AddOutput("c->out", "gpu"));
  b.AddOutput("b->out_gpu");
  b.AddOutput("c->out_gpu");
  auto g = std::move(b).GetGraph(true);
  ExecGraph eg;
  eg.Lower(g);

  StreamAssignment<StreamPolicy::PerBackend> assignment(eg);
  auto map = MakeNodeMap(eg);
  EXPECT_EQ(assignment[map["a"]], std::nullopt);
  EXPECT_EQ(assignment[map["b"]], 1);
  EXPECT_EQ(assignment[map["c"]], 0);
}

}  // namespace exec2
}  // namespace dali
