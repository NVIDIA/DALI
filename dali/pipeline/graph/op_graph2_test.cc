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
#include "dali/core/int_literals.h"
#include "dali/pipeline/graph/op_graph2.h"

namespace dali {
namespace graph {
namespace test {

TEST(NewOpGraphTest, AddEraseOp) {
  OpGraph g;
  OpSpec spec("dummy");
  OpSpec otherspec("asdf");
  OpNode &op_node = g.AddOp("instance1", spec);
  EXPECT_THROW(g.AddOp("instance1", otherspec), std::invalid_argument);
  EXPECT_EQ(op_node.instance_name, "instance1");
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
  spec.AddInput("i0",  "cpu");
  spec.AddInput("i1",  "gpu");
  spec.AddOutput("o0", "gpu");
  spec.AddOutput("o1", "cpu");

  OpGraph::Builder b;
  b.Add("op1", spec);
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
}


TEST(NewOpGraphBuilderTest, AddMultipleOps) {
  OpSpec spec("dummy");
  spec.AddInput("i0",  "cpu");
  spec.AddInput("i1",  "gpu");
  spec.AddOutput("o0", "gpu");
  spec.AddOutput("o1", "cpu");

  OpGraph::Builder b;
  b.Add("op1", spec);
  b.Add("op2", spec);
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
}


}  // namespace test
}  // namespace graph
}  // namespace dali
