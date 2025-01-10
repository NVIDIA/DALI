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

#include <gtest/gtest.h>
#include "dali/core/int_literals.h"
#include "dali/pipeline/graph/op_graph2.h"

namespace dali {
namespace graph {
namespace test {

TEST(NewOpGraphTest, AddEraseOp) {
  OpGraph g;
  OpSpec spec("dummy");
  spec.AddArg("device", "gpu");
  OpSpec otherspec("asdf");
  OpNode &op_node = g.AddOp("instance1", spec);
  EXPECT_THROW(g.AddOp("instance1", otherspec), std::invalid_argument);
  EXPECT_EQ(op_node.instance_name, "instance1");
  EXPECT_EQ(op_node.op_type, OpType::GPU) << "The spec's device not parsed properly.";
  EXPECT_EQ(op_node.spec.SchemaName(), "dummy");
  EXPECT_EQ(g.GetOp("instance1"), &op_node);
  EXPECT_TRUE(g.EraseOp("instance1"));
  EXPECT_EQ(g.GetOp("instance1"), nullptr);
}

TEST(NewOpGraphTest, AddEraseData) {
  OpGraph g;
  auto dev = StorageDevice::CPU;
  DataNode &data_node = g.AddData("d1", dev);
  EXPECT_THROW(g.AddData("d1", StorageDevice::GPU), std::invalid_argument);
  EXPECT_EQ(data_node.name, "d1");
  EXPECT_EQ(data_node.device, dev);
  EXPECT_EQ(g.GetData("d1"), &data_node);
  EXPECT_TRUE(g.EraseData("d1"));
  EXPECT_EQ(g.GetData("d1"), nullptr);
}

TEST(NewOpGraphBuilderTest, AddSingleOp) {
  OpSpec spec("dummy");
  spec.AddInput("i0",  StorageDevice::CPU);
  spec.AddInput("i1",  StorageDevice::GPU);
  spec.AddOutput("o0", StorageDevice::GPU);
  spec.AddOutput("o1", StorageDevice::CPU);

  OpGraph::Builder b;
  b.Add("op1", spec);
  b.AddOutput("o0_gpu");
  b.AddOutput("o1_cpu");
  b.AddOutput("o0_gpu");
  b.AddOutput("o1_cpu");
  OpGraph g = std::move(b).GetGraph();
  auto *op1 = g.GetOp("op1");
  ASSERT_NE(op1, nullptr);
  EXPECT_EQ(op1->instance_name, "op1");
  EXPECT_EQ(op1->spec.SchemaName(), "dummy");


  DataNode *o0 = g.GetData("o0_gpu");
  ASSERT_NE(o0, nullptr);
  EXPECT_EQ(o0->device, StorageDevice::GPU);
  EXPECT_EQ(o0->producer.op, op1);
  EXPECT_EQ(o0->producer.idx, 0);
  EXPECT_EQ(o0->consumers.size(), 0_uz);

  DataNode *o1 = g.GetData("o1_cpu");
  ASSERT_NE(o1, nullptr);
  EXPECT_EQ(o1->device, StorageDevice::CPU);
  EXPECT_EQ(o1->producer.op, op1);
  EXPECT_EQ(o1->producer.idx, 1);
  EXPECT_EQ(o1->consumers.size(), 0_uz);


  DataNode *i0 = g.GetData("i0_cpu");
  ASSERT_NE(i0, nullptr);
  EXPECT_EQ(i0->device, StorageDevice::CPU);
  EXPECT_EQ(i0->producer.op, nullptr);
  ASSERT_EQ(i0->consumers.size(), 1_uz);
  EXPECT_EQ(i0->consumers[0].op, op1);
  EXPECT_EQ(i0->consumers[0].idx, 0);

  DataNode *i1 = g.GetData("i1_gpu");
  ASSERT_NE(i1, nullptr);
  EXPECT_EQ(i1->device, StorageDevice::GPU);
  EXPECT_EQ(i1->producer.op, nullptr);
  ASSERT_EQ(i1->consumers.size(), 1_uz);
  EXPECT_EQ(i1->consumers[0].op, op1);
  EXPECT_EQ(i1->consumers[0].idx, 1);

  ASSERT_EQ(op1->inputs.size(), 2_uz);
  ASSERT_EQ(op1->outputs.size(), 2_uz);
  EXPECT_EQ(op1->outputs[0], o0);
  EXPECT_EQ(op1->outputs[1], o1);
  EXPECT_EQ(op1->inputs[0], i0);
  EXPECT_EQ(op1->inputs[1], i1);
}


TEST(NewOpGraphBuilderTest, AddMultipleOps) {
  /*
  Graph topology

  i0   i1
   \  /   \
    op1    \
   /   \    |
  m0    m1  |
   \   /    |
    \ /     |
     /     /
    / \   /
   (   ) /
    op2
   /   \
  o0    o1
  |     |
  v     v (pipeline outputs)
  */

  OpSpec spec1("dummy");
  spec1.AddInput("i0",  StorageDevice::CPU);
  spec1.AddInput("i1",  StorageDevice::GPU);
  spec1.AddOutput("m0", StorageDevice::GPU);
  spec1.AddOutput("m1", StorageDevice::CPU);

  OpSpec spec2("dummy");
  spec2.AddArg("device", "mixed");
  spec2.AddInput("m1",  StorageDevice::CPU);
  spec2.AddInput("m0",  StorageDevice::GPU);
  spec2.AddInput("i1",  StorageDevice::GPU);
  spec2.AddOutput("o0", StorageDevice::GPU);
  spec2.AddOutput("o1", StorageDevice::CPU);


  OpGraph::Builder b;
  b.Add("op1", spec1);
  b.Add("op2", spec2);
  b.AddOutput("o0_gpu");
  b.AddOutput("o1_cpu");
  OpGraph g = std::move(b).GetGraph();
  auto *op1 = g.GetOp("op1");
  auto *op2 = g.GetOp("op2");
  ASSERT_NE(op1, nullptr);
  EXPECT_EQ(op1->op_type, OpType::CPU);
  EXPECT_EQ(op1->instance_name, "op1");
  EXPECT_EQ(op1->spec.SchemaName(), "dummy");

  ASSERT_NE(op2, nullptr);
  EXPECT_EQ(op2->op_type, OpType::MIXED);
  EXPECT_EQ(op2->instance_name, "op2");
  EXPECT_EQ(op2->spec.SchemaName(), "dummy");


  DataNode *i0 = g.GetData("i0_cpu");
  ASSERT_NE(i0, nullptr);
  EXPECT_EQ(i0->device, StorageDevice::CPU);
  EXPECT_EQ(i0->producer.op, nullptr);
  ASSERT_EQ(i0->consumers.size(), 1_uz);
  EXPECT_EQ(i0->consumers[0].op, op1);
  EXPECT_EQ(i0->consumers[0].idx, 0);

  DataNode *i1 = g.GetData("i1_gpu");
  ASSERT_NE(i1, nullptr);
  EXPECT_EQ(i1->device, StorageDevice::GPU);
  EXPECT_EQ(i1->producer.op, nullptr);
  ASSERT_EQ(i1->consumers.size(), 2_uz);
  EXPECT_EQ(i1->consumers[0].op, op1);
  EXPECT_EQ(i1->consumers[0].idx, 1);
  EXPECT_EQ(i1->consumers[1].op, op2);
  EXPECT_EQ(i1->consumers[1].idx, 2);

  DataNode *m0 = g.GetData("m0_gpu");
  ASSERT_NE(m0, nullptr);
  EXPECT_EQ(m0->device, StorageDevice::GPU);
  EXPECT_EQ(m0->producer.op, op1);
  EXPECT_EQ(m0->producer.idx, 0);
  ASSERT_EQ(m0->consumers.size(), 1_uz);
  EXPECT_EQ(m0->consumers[0].op, op2);
  EXPECT_EQ(m0->consumers[0].idx, 1);

  DataNode *m1 = g.GetData("m1_cpu");
  ASSERT_NE(m1, nullptr);
  EXPECT_EQ(m1->device, StorageDevice::CPU);
  EXPECT_EQ(m1->producer.op, op1);
  EXPECT_EQ(m1->producer.idx, 1);
  ASSERT_EQ(m1->consumers.size(), 1_uz);
  EXPECT_EQ(m1->consumers[0].op, op2);
  EXPECT_EQ(m1->consumers[0].idx, 0);

  DataNode *o0 = g.GetData("o0_gpu");
  ASSERT_NE(o0, nullptr);
  EXPECT_EQ(o0->device, StorageDevice::GPU);
  EXPECT_EQ(o0->producer.op, op2);
  EXPECT_EQ(o0->producer.idx, 0);
  EXPECT_EQ(o0->consumers.size(), 0_uz);

  DataNode *o1 = g.GetData("o1_cpu");
  ASSERT_NE(o1, nullptr);
  EXPECT_EQ(o1->device, StorageDevice::CPU);
  EXPECT_EQ(o1->producer.op, op2);
  EXPECT_EQ(o1->producer.idx, 1);
  EXPECT_EQ(o1->consumers.size(), 0_uz);

  ASSERT_EQ(op1->inputs.size(), 2_uz);
  ASSERT_EQ(op1->outputs.size(), 2_uz);
  EXPECT_EQ(op1->inputs[0], i0);
  EXPECT_EQ(op1->inputs[1], i1);
  EXPECT_EQ(op1->outputs[0], m0);
  EXPECT_EQ(op1->outputs[1], m1);

  ASSERT_EQ(op2->inputs.size(), 3_uz);
  ASSERT_EQ(op2->outputs.size(), 2_uz);
  EXPECT_EQ(op2->inputs[0], m1);
  EXPECT_EQ(op2->inputs[1], m0);
  EXPECT_EQ(op2->inputs[2], i1);
  EXPECT_EQ(op2->outputs[0], o0);
  EXPECT_EQ(op2->outputs[1], o1);
}


TEST(NewOpGraphBuilderTest, SortAndPrune) {
  /*
  Graph topology

  i0   i1
   \  /   \
    op1    \
   /   \    |
  m0    m1  |
   \   / \  |
    \ /   \ |
     /     /
    / \   / \
   (   ) /   \
    op2      op3      op4 (preserved)
   /   \      |
  o0    o1    o2
  |     |     |
  |     |     x (unused)
  |     |
  v     v (pipeline outputs)
  */

  OpSpec spec1("dummy");
  spec1.AddArg("device", "cpu");
  spec1.AddInput("i0",  StorageDevice::CPU);
  spec1.AddInput("i1",  StorageDevice::GPU);
  spec1.AddOutput("m0", StorageDevice::GPU);
  spec1.AddOutput("m1", StorageDevice::CPU);

  OpSpec spec2("dummy");
  spec2.AddArg("device", "gpu");
  spec2.AddInput("m1",  StorageDevice::CPU);
  spec2.AddInput("m0",  StorageDevice::GPU);
  spec2.AddInput("i1",  StorageDevice::GPU);
  spec2.AddOutput("o0", StorageDevice::GPU);
  spec2.AddOutput("o1", StorageDevice::CPU);

  OpSpec spec3("dummy");
  spec3.AddInput("m1",  StorageDevice::CPU);
  spec3.AddInput("i1",  StorageDevice::GPU);
  spec3.AddOutput("o2", StorageDevice::GPU);

  OpSpec spec4("dummy");
  spec4.AddArg("preserve", true);

  OpGraph::Builder b;
  b.Add("op2", spec2);
  b.Add("op3", spec3);
  b.Add("op1", spec1);
  b.Add("op4", spec4);
  b.AddOutput("o0_gpu");
  b.AddOutput("o1_cpu");
  OpGraph g = std::move(b).GetGraph(false);  // don't prune now

  auto &nodes = g.OpNodes();
  ASSERT_EQ(nodes.size(), 4_uz) << "The nodes were not added properly - abandoning test";
  g.Sort(true);
  EXPECT_EQ(nodes.size(), 3_uz);

  auto *op1 = g.GetOp("op1");
  auto *op2 = g.GetOp("op2");
  auto *op3 = g.GetOp("op3");
  auto *op4 = g.GetOp("op4");
  EXPECT_EQ(op3, nullptr) << "The operator op3 should have been pruned";
  EXPECT_NE(op4, nullptr) << "The operator op4 should NOT have been pruned";

  ASSERT_NE(op1, nullptr) << "Operator op1 not found in the pruned graph";
  EXPECT_EQ(op1->op_type, OpType::CPU);
  EXPECT_EQ(op1->instance_name, "op1");
  EXPECT_EQ(op1->spec.SchemaName(), "dummy");

  ASSERT_NE(op2, nullptr) << "Operator op2 not found in the pruned graph";
  EXPECT_EQ(op2->op_type, OpType::GPU);
  EXPECT_EQ(op2->instance_name, "op2");
  EXPECT_EQ(op2->spec.SchemaName(), "dummy");


  DataNode *o2 = g.GetData("o2_gpu");
  EXPECT_EQ(o2, nullptr) << "The data node o2 should have been pruned";


  DataNode *i0 = g.GetData("i0_cpu");
  ASSERT_NE(i0, nullptr);
  EXPECT_EQ(i0->device, StorageDevice::CPU);
  EXPECT_EQ(i0->producer.op, nullptr);
  ASSERT_EQ(i0->consumers.size(), 1_uz);
  EXPECT_EQ(i0->consumers[0].op, op1);
  EXPECT_EQ(i0->consumers[0].idx, 0);

  DataNode *i1 = g.GetData("i1_gpu");
  ASSERT_NE(i1, nullptr);
  EXPECT_EQ(i1->device, StorageDevice::GPU);
  EXPECT_EQ(i1->producer.op, nullptr);
  ASSERT_EQ(i1->consumers.size(), 2_uz);
  EXPECT_EQ(i1->consumers[0].op, op2);
  EXPECT_EQ(i1->consumers[0].idx, 2);
  EXPECT_EQ(i1->consumers[1].op, op1);
  EXPECT_EQ(i1->consumers[1].idx, 1);

  DataNode *m0 = g.GetData("m0_gpu");
  ASSERT_NE(m0, nullptr);
  EXPECT_EQ(m0->device, StorageDevice::GPU);
  EXPECT_EQ(m0->producer.op, op1);
  EXPECT_EQ(m0->producer.idx, 0);
  ASSERT_EQ(m0->consumers.size(), 1_uz);
  EXPECT_EQ(m0->consumers[0].op, op2);
  EXPECT_EQ(m0->consumers[0].idx, 1);

  DataNode *m1 = g.GetData("m1_cpu");
  ASSERT_NE(m1, nullptr);
  EXPECT_EQ(m1->device, StorageDevice::CPU);
  EXPECT_EQ(m1->producer.op, op1);
  EXPECT_EQ(m1->producer.idx, 1);
  ASSERT_EQ(m1->consumers.size(), 1_uz);
  EXPECT_EQ(m1->consumers[0].op, op2);
  EXPECT_EQ(m1->consumers[0].idx, 0);

  DataNode *o0 = g.GetData("o0_gpu");
  ASSERT_NE(o0, nullptr);
  EXPECT_EQ(o0->device, StorageDevice::GPU);
  EXPECT_EQ(o0->producer.op, op2);
  EXPECT_EQ(o0->producer.idx, 0);
  EXPECT_EQ(o0->consumers.size(), 0_uz);

  DataNode *o1 = g.GetData("o1_cpu");
  ASSERT_NE(o1, nullptr);
  EXPECT_EQ(o1->device, StorageDevice::CPU);
  EXPECT_EQ(o1->producer.op, op2);
  EXPECT_EQ(o1->producer.idx, 1);
  EXPECT_EQ(o1->consumers.size(), 0_uz);

  ASSERT_EQ(op1->inputs.size(), 2_uz);
  ASSERT_EQ(op1->outputs.size(), 2_uz);
  EXPECT_EQ(op1->inputs[0], i0);
  EXPECT_EQ(op1->inputs[1], i1);
  EXPECT_EQ(op1->outputs[0], m0);
  EXPECT_EQ(op1->outputs[1], m1);

  ASSERT_EQ(op2->inputs.size(), 3_uz);
  ASSERT_EQ(op2->outputs.size(), 2_uz);
  EXPECT_EQ(op2->inputs[0], m1);
  EXPECT_EQ(op2->inputs[1], m0);
  EXPECT_EQ(op2->inputs[2], i1);
  EXPECT_EQ(op2->outputs[0], o0);
  EXPECT_EQ(op2->outputs[1], o1);
}

}  // namespace test
}  // namespace graph
}  // namespace dali
