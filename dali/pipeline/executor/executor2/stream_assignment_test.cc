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

  bool HasContiguousOutputs() const override {
    return false;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return false;
  }
};

DALI_SCHEMA(StreamAssignmentDummyOp)
  .NumInput(0, 999)
  .InputDevice(0, 999, InputDevice::Any)
  .NumOutput(0)
  .AdditionalOutputsFn([](const OpSpec &spec) {
    return spec.NumOutput();
  });

DALI_REGISTER_OPERATOR(StreamAssignmentDummyOp, StreamAssignmentDummyOp<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(StreamAssignmentDummyOp, StreamAssignmentDummyOp<MixedBackend>, Mixed);
DALI_REGISTER_OPERATOR(StreamAssignmentDummyOp, StreamAssignmentDummyOp<GPUBackend>, GPU);


template <typename Backend>
class StreamAssignmentMetaOp : public Operator<Backend> {
 public:
  using Operator<Backend>::Operator;
  USE_OPERATOR_MEMBERS();

  void RunImpl(Workspace &ws) override {}
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return false;
  }
};

DALI_SCHEMA(StreamAssignmentMetaOp)
  .NumInput(0, 999)
  .InputDevice(0, 999, InputDevice::Metadata)
  .NumOutput(0)
  .AdditionalOutputsFn([](const OpSpec &spec) {
    return spec.NumOutput();
  });

DALI_REGISTER_OPERATOR(StreamAssignmentMetaOp, StreamAssignmentMetaOp<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(StreamAssignmentMetaOp, StreamAssignmentMetaOp<MixedBackend>, Mixed);
DALI_REGISTER_OPERATOR(StreamAssignmentMetaOp, StreamAssignmentMetaOp<GPUBackend>, GPU);

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


OpSpec SpecMetaDev(const std::string &device) {
  return OpSpec("StreamAssignmentMetaOp")
    .AddArg("device", device)
    .AddArg("num_threads", 1)
    .AddArg("max_batch_size", 1);
}

OpSpec SpecMetaGPU() {
  return SpecMetaDev("gpu");
}

OpSpec SpecMetaCPU() {
  return SpecMetaDev("cpu");
}

OpSpec SpecMetaMixed() {
  return SpecMetaDev("mixed");
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

template <StreamPolicy policy>
void TestGPU2CPUAssignment() {
      graph::OpGraph::Builder b;
  b.Add("a",
        SpecGPU()
        .AddOutput("a->b", "gpu")
        .AddOutput("a->c", "gpu"));
  b.Add("b",
        SpecCPU()
        .AddInput("a->b", "gpu")
        .AddOutput("b->out", "cpu"));
  b.Add("c",
        SpecMetaCPU()
        .AddInput("a->c", "gpu")
        .AddOutput("c->out", "cpu"));
  b.AddOutput("b->out_cpu");
  b.AddOutput("c->out_cpu");
  auto g = std::move(b).GetGraph(true);
  ExecGraph eg;
  eg.Lower(g);

  StreamAssignment<policy> assignment(eg);
  auto map = MakeNodeMap(eg);
  EXPECT_EQ(assignment[map["a"]], 0);
  EXPECT_EQ(assignment[map["b"]], 0);  // CPU operator with GPU input
  EXPECT_EQ(assignment[map["c"]], std::nullopt);  // metadata-only
}

TEST(Exec2Test, StreamAssignment_Single_GPU2CPU) {
  TestGPU2CPUAssignment<StreamPolicy::Single>();
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


TEST(Exec2Test, StreamAssignment_PerBackend_GPU2CPU) {
  TestGPU2CPUAssignment<StreamPolicy::PerBackend>();
}

TEST(Exec2Test, StreamAssignment_OperOperator_GPU2CPU) {
  TestGPU2CPUAssignment<StreamPolicy::PerOperator>();
}

TEST(Exec2Test, StreamAssignment_PerOperator_1) {
  ExecGraph eg;
  /*
  a -- b ----- c --------- g --  out
   \                      /
    ---d -- e (cpu)---f (mixed)

  */
  graph::OpGraph::Builder b;
  b.Add("a",
        SpecGPU()
        .AddOutput("a->b", "gpu")
        .AddOutput("a->d", "gpu"));
  b.Add("b",
        SpecGPU()
        .AddInput("a->b", "gpu")
        .AddOutput("b->c", "gpu"));
  b.Add("c",
        SpecGPU()
        .AddInput("b->c", "gpu")
        .AddOutput("c->g", "gpu"));
  b.Add("d",
        SpecGPU()
        .AddInput("a->d", "gpu")
        .AddOutput("d->e", "cpu"));
  b.Add("e",
        SpecCPU()
        .AddInput("d->e", "cpu")
        .AddOutput("e->f", "cpu"));
  b.Add("f",
        SpecMixed()
        .AddInput("e->f", "cpu")
        .AddOutput("f->g", "gpu"));
  b.Add("g",
        SpecGPU()
        .AddInput("c->g", "gpu")
        .AddInput("f->g", "gpu")
        .AddOutput("g->o", "gpu"));
  b.AddOutput("g->o_gpu");
  auto g = std::move(b).GetGraph(true);
  eg.Lower(g);

  StreamAssignment<StreamPolicy::PerOperator> assignment(eg);
  auto map = MakeNodeMap(eg);
  EXPECT_EQ(assignment[map["a"]], 0);
  EXPECT_EQ(assignment[map["b"]], 0);
  EXPECT_EQ(assignment[map["c"]], 0);
  EXPECT_EQ(assignment[map["d"]], 1);
  EXPECT_EQ(assignment[map["e"]], std::nullopt);
  EXPECT_EQ(assignment[map["f"]], 1);
  EXPECT_EQ(assignment[map["g"]], 0);
}


TEST(Exec2Test, StreamAssignment_PerOperator_2) {
  ExecGraph eg;
  /*
         --c--             g
        /     \         /   \
  a -- b ----- d ----- f ---- h ---> out
   \   (cpu)       /       /
    --------------e       /
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
        SpecMetaCPU()
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
  EXPECT_EQ(assignment[map["b"]], 0);  // CPU operator with a GPU input needs a stream
  EXPECT_EQ(assignment[map["c"]], 0);
  EXPECT_EQ(assignment[map["d"]], 0);
  EXPECT_EQ(assignment[map["e"]], 1);
  EXPECT_EQ(assignment[map["f"]], 0);
  EXPECT_EQ(assignment[map["g"]], 0);
  EXPECT_EQ(assignment[map["h"]], 0);
  EXPECT_EQ(assignment[map["i"]], 2);
  EXPECT_EQ(assignment[map["j"]], std::nullopt);  // metadata only
  EXPECT_EQ(assignment[map["k"]], 3);
}


TEST(Exec2Test, StreamAssignment_PerOperator_3) {
  ExecGraph eg;
  /*
  a --- c --- e -- out
    \ /   \ /
     X     X
    / \   / \
  b --- d --- f -- out
  */
  graph::OpGraph::Builder b;
  b.Add("a",
        SpecGPU()
        .AddOutput("a->c", "gpu")
        .AddOutput("a->d", "gpu"));
  b.Add("b",
        SpecGPU()
        .AddOutput("b->c", "gpu")
        .AddOutput("b->d", "gpu"));
  b.Add("c",
        SpecGPU()
        .AddInput("a->c", "gpu")
        .AddInput("b->c", "gpu")
        .AddOutput("c->e", "gpu")
        .AddOutput("c->f", "gpu"));
  b.Add("d",
        SpecGPU()
        .AddInput("a->d", "gpu")
        .AddInput("b->d", "gpu")
        .AddOutput("d->e", "gpu")
        .AddOutput("d->f", "gpu"));
  b.Add("e",
        SpecGPU()
        .AddInput("c->e", "gpu")
        .AddInput("d->e", "gpu")
        .AddOutput("e->o", "gpu"));
  b.Add("f",
        SpecGPU()
        .AddInput("c->f", "gpu")
        .AddInput("d->f", "gpu")
        .AddOutput("f->o", "gpu"));
  b.AddOutput("e->o_gpu");
  b.AddOutput("f->o_gpu");
  auto g = std::move(b).GetGraph(true);
  eg.Lower(g);

  StreamAssignment<StreamPolicy::PerOperator> assignment(eg);
  auto map = MakeNodeMap(eg);
  EXPECT_EQ(assignment[map["a"]], 0);
  EXPECT_EQ(assignment[map["b"]], 1);
  EXPECT_EQ(assignment[map["c"]], 0);
  EXPECT_EQ(assignment[map["d"]], 1);
  EXPECT_EQ(assignment[map["e"]], 0);
  EXPECT_EQ(assignment[map["f"]], 1);
}

TEST(Exec2Test, StreamAssignment_PerOperator_4) {
  ExecGraph eg;
  /*
  a --- b -- out

  c (sink node)
  d (sink node)
  */
  graph::OpGraph::Builder b;
  b.Add("a",
        SpecGPU()
        .AddOutput("a->b", "gpu"));
  b.Add("b",
        SpecGPU()
        .AddInput("a->b", "gpu")
        .AddOutput("b->o", "gpu"));
  b.Add("c",
        SpecGPU()
        .AddArg("preserve", true));
  b.Add("d",
        SpecGPU()
        .AddArg("preserve", true));
  b.AddOutput("b->o_gpu");
  auto g = std::move(b).GetGraph(true);
  eg.Lower(g);

  StreamAssignment<StreamPolicy::PerOperator> assignment(eg);
  auto map = MakeNodeMap(eg);
  EXPECT_EQ(assignment[map["a"]], 0);
  EXPECT_EQ(assignment[map["b"]], 0);
  EXPECT_EQ(assignment[map["c"]], 1);
  EXPECT_EQ(assignment[map["d"]], 2);
}

TEST(Exec2Test, StreamAssignment_PerOperator_5) {
  ExecGraph eg;
  graph::OpGraph::Builder b;
  b.Add("a",
        SpecGPU()
        .AddOutput("a->d", "gpu")
        .AddOutput("a->e", "gpu")
        .AddOutput("a->f", "gpu")
        .AddOutput("a->c", "gpu"));
  b.Add("b",
        SpecGPU()
        .AddOutput("b->d", "gpu")
        .AddOutput("b->e", "gpu")
        .AddOutput("b->f", "gpu")
        .AddOutput("b->c", "gpu"));
  b.Add("c",
        SpecGPU()
        .AddInput("a->c", "gpu")
        .AddInput("b->c", "gpu")
        .AddOutput("c->g", "gpu"));
  b.Add("d",
        SpecGPU()
        .AddInput("a->d", "gpu")
        .AddInput("b->d", "gpu")
        .AddOutput("d->od", "gpu"));
  b.Add("e",
        SpecGPU()
        .AddInput("a->e", "gpu")
        .AddInput("b->e", "gpu")
        .AddOutput("e->oe", "gpu"));
  b.Add("f",
        SpecGPU()
        .AddInput("a->f", "gpu")
        .AddInput("b->f", "gpu")
        .AddOutput("f->of", "gpu"));
  b.Add("g",
        SpecGPU()
        .AddInput("c->g", "gpu")
        .AddOutput("g->og", "gpu"));
  b.AddOutput("d->od_gpu");
  b.AddOutput("e->oe_gpu");
  b.AddOutput("f->of_gpu");
  b.AddOutput("g->og_gpu");
  auto g = std::move(b).GetGraph(true);
  eg.Lower(g);

  StreamAssignment<StreamPolicy::PerOperator> assignment(eg);
  auto map = MakeNodeMap(eg);
  EXPECT_EQ(assignment[map["a"]], 0);
  EXPECT_EQ(assignment[map["b"]], 1);
  EXPECT_EQ(assignment[map["c"]], 3);
  EXPECT_EQ(assignment[map["d"]], 0);
  EXPECT_EQ(assignment[map["e"]], 1);
  EXPECT_EQ(assignment[map["f"]], 2);
  EXPECT_EQ(assignment[map["g"]], 3);
}


TEST(Exec2Test, StreamAssignment_PerOperator_6) {
  ExecGraph eg;
  graph::OpGraph::Builder b;
  b.Add("a",
        SpecGPU()
        .AddOutput("a->d", "gpu")
        .AddOutput("a->e", "gpu")
        .AddOutput("a->f", "gpu")
        .AddOutput("a->c", "gpu"));
  b.Add("b",
        SpecGPU()
        .AddOutput("b->d", "gpu")
        .AddOutput("b->e", "gpu")
        .AddOutput("b->f", "gpu")
        .AddOutput("b->c", "gpu"));
  b.Add("c",
        SpecGPU()
        .AddInput("a->c", "gpu")
        .AddInput("b->c", "gpu")
        .AddOutput("c->g", "gpu"));
  b.Add("d",
        SpecGPU()
        .AddInput("a->d", "gpu")
        .AddInput("b->d", "gpu")
        .AddOutput("d->od", "gpu"));
  b.Add("e",
        SpecGPU()
        .AddInput("a->e", "gpu")
        .AddInput("b->e", "gpu")
        .AddOutput("e->oe", "gpu"));
  b.Add("f",
        SpecGPU()
        .AddInput("a->f", "gpu")
        .AddInput("b->f", "gpu")
        .AddOutput("f->of", "gpu"));
  b.Add("g",
        SpecGPU()
        .AddInput("c->g", "gpu")
        .AddOutput("g->og", "gpu"));
  b.AddOutput("d->od_gpu");
  b.AddOutput("e->oe_gpu");
  b.AddOutput("f->of_gpu");
  b.AddOutput("g->og_gpu");
  auto g = std::move(b).GetGraph(true);
  eg.Lower(g);

  StreamAssignment<StreamPolicy::PerOperator> assignment(eg);
  auto map = MakeNodeMap(eg);
  EXPECT_EQ(assignment[map["a"]], 0);
  EXPECT_EQ(assignment[map["b"]], 1);
  EXPECT_EQ(assignment[map["c"]], 3);
  EXPECT_EQ(assignment[map["d"]], 0);
  EXPECT_EQ(assignment[map["e"]], 1);
  EXPECT_EQ(assignment[map["f"]], 2);
  EXPECT_EQ(assignment[map["g"]], 3);
}


TEST(Exec2Test, StreamAssignment_PerOperator_MultiBranch) {
  /*
    b   e
   / \ / \
  a   d   g --- out
   \ / \ /
    c   f

    i   l
   / \ / \
  h   k   n --- out
   \ / \ /
    j   m
  */
  ExecGraph eg;
  graph::OpGraph::Builder b;
  b.Add("a",
        SpecGPU()
        .AddOutput("a->b", "gpu")
        .AddOutput("a->c", "gpu"));
  b.Add("b",
        SpecGPU()
        .AddInput("a->b", "gpu")
        .AddOutput("b->d", "gpu"));
  b.Add("c",
        SpecGPU()
        .AddInput("a->c", "gpu")
        .AddOutput("c->d", "gpu"));
  b.Add("d",
        SpecGPU()
        .AddInput("b->d", "gpu")
        .AddInput("c->d", "gpu")
        .AddOutput("d->e", "gpu")
        .AddOutput("d->f", "gpu"));
  b.Add("e",
        SpecGPU()
        .AddInput("d->e", "gpu")
        .AddOutput("e->g", "gpu"));
  b.Add("f",
        SpecGPU()
        .AddInput("d->f", "gpu")
        .AddOutput("f->g", "gpu"));
  b.Add("g",
        SpecGPU()
        .AddInput("e->g", "gpu")
        .AddInput("f->g", "gpu")
        .AddOutput("g->og", "gpu"));

  b.Add("h",
        SpecGPU()
        .AddOutput("h->i", "gpu")
        .AddOutput("h->j", "gpu"));
  b.Add("i",
        SpecGPU()
        .AddInput("h->i", "gpu")
        .AddOutput("i->k", "gpu"));
  b.Add("j",
        SpecGPU()
        .AddInput("h->j", "gpu")
        .AddOutput("j->k", "gpu"));
  b.Add("k",
        SpecGPU()
        .AddInput("i->k", "gpu")
        .AddInput("j->k", "gpu")
        .AddOutput("k->l", "gpu")
        .AddOutput("k->m", "gpu"));
  b.Add("l",
        SpecGPU()
        .AddInput("k->l", "gpu")
        .AddOutput("l->n", "gpu"));
  b.Add("m",
        SpecGPU()
        .AddInput("k->m", "gpu")
        .AddOutput("m->n", "gpu"));
  b.Add("n",
        SpecGPU()
        .AddInput("l->n", "gpu")
        .AddInput("m->n", "gpu")
        .AddOutput("n->on", "gpu"));

  b.AddOutput("g->og_gpu");
  b.AddOutput("n->on_gpu");
  auto g = std::move(b).GetGraph(true);
  eg.Lower(g);

  StreamAssignment<StreamPolicy::PerOperator> assignment(eg);
  auto map = MakeNodeMap(eg);
  EXPECT_EQ(assignment[map["a"]], 0);
  EXPECT_EQ(assignment[map["b"]], 0);
  EXPECT_EQ(assignment[map["c"]], 1);
  EXPECT_EQ(assignment[map["d"]], 0);
  EXPECT_EQ(assignment[map["e"]], 0);
  EXPECT_EQ(assignment[map["f"]], 1);
  EXPECT_EQ(assignment[map["g"]], 0);

  EXPECT_EQ(assignment[map["h"]], 2);
  EXPECT_EQ(assignment[map["i"]], 2);
  EXPECT_EQ(assignment[map["j"]], 3);
  EXPECT_EQ(assignment[map["k"]], 2);
  EXPECT_EQ(assignment[map["l"]], 2);
  EXPECT_EQ(assignment[map["m"]], 3);
  EXPECT_EQ(assignment[map["n"]], 2);
}


TEST(Exec2Test, StreamAssignment_PerOperator_BranchOut) {
  /*
  a   d --- out
   \ /
    c
   / \
  b   e --- out
   \
    f ------ out
  */
  ExecGraph eg;
  graph::OpGraph::Builder b;
  b.Add("a",
        SpecGPU()
        .AddOutput("a->c", "gpu"));
  b.Add("b",
        SpecGPU()
        .AddOutput("b->c", "gpu")
        .AddOutput("b->f", "gpu"));
  b.Add("c",
        SpecGPU()
        .AddInput("a->c", "gpu")
        .AddInput("b->c", "gpu")
        .AddOutput("c->d", "gpu")
        .AddOutput("c->e", "gpu"));
  b.Add("d",
        SpecGPU()
        .AddInput("c->d", "gpu")
        .AddOutput("d->o", "gpu"));
  b.Add("e",
        SpecGPU()
        .AddInput("c->e", "gpu")
        .AddOutput("e->o", "gpu"));
  b.Add("f",
        SpecGPU()
        .AddInput("b->f", "gpu")
        .AddOutput("f->o", "gpu"));

  b.AddOutput("d->o_gpu");
  b.AddOutput("e->o_gpu");
  b.AddOutput("f->o_gpu");
  auto g = std::move(b).GetGraph(true);
  eg.Lower(g);

  StreamAssignment<StreamPolicy::PerOperator> assignment(eg);
  auto map = MakeNodeMap(eg);
  EXPECT_EQ(assignment[map["a"]], 0);
  EXPECT_EQ(assignment[map["b"]], 1);
  EXPECT_EQ(assignment[map["c"]], 0);
  EXPECT_EQ(assignment[map["d"]], 0);
  EXPECT_EQ(assignment[map["e"]], 1);
  EXPECT_EQ(assignment[map["f"]], 2);
}

}  // namespace exec2
}  // namespace dali
