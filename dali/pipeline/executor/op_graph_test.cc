// Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <stdexcept>

#include "dali/pipeline/executor/lowered_graph.h"
#include "dali/test/dali_test.h"

namespace dali {

class OpGraphTest : public DALITest {
 public:
  inline OpSpec& PrepareSpec(OpSpec &spec) {
    spec.AddArg("max_batch_size", 1)
      .AddArg("num_threads", 1)
      .AddArg("cuda_stream", 0)
      .AddArg("pixels_per_image_hint", 0);
    return spec;
  }
};

TEST_F(OpGraphTest, TestCPUOnly) {
  OpGraph graph;

  // Add copy op insertion
  // Add contiguous-ify op
  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "cpu")
          .AddOutput("external_data", StorageDevice::CPU)), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("Copy")
          .AddInput("external_data", StorageDevice::CPU)
          .AddOutput("copy_data", StorageDevice::CPU)), "");

  // Validate the graph
  ASSERT_EQ(graph.NumOp(OpType::CPU), 2);
  ASSERT_EQ(graph.NumOp(OpType::MIXED), 0);
  ASSERT_EQ(graph.NumOp(OpType::GPU), 0);
  ASSERT_EQ(graph.NumTensor(), 2);

  // Validate the source op
  auto& node = graph.Node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 1);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(node.children.count(1), 1);
  ASSERT_EQ(graph.TensorSourceID(node.spec.Output(0)), 0);
  ASSERT_EQ(graph.TensorIdxInSource(node.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node.spec.Output(0)));
  ASSERT_EQ(graph.Tensor(0).name, "external_data_cpu");
  ASSERT_EQ(graph.Tensor(0).producer.node, 0);
  ASSERT_EQ(graph.Tensor(0).consumers.size(), 1);
  ASSERT_EQ(graph.Tensor(0).consumers[0].node, 1);
  ASSERT_EQ(node.parent_tensors.size(), 0);
  ASSERT_EQ(node.children_tensors.size(), 1);
  ASSERT_EQ(node.children_tensors[0], 0);

  // Validate copy op
  auto& node2 = graph.Node(1);
  ASSERT_EQ(node2.id, 1);
  ASSERT_EQ(node2.children.size(), 0);
  ASSERT_EQ(node2.parents.size(), 1);
  ASSERT_EQ(node2.parents.count(0), 1);
  ASSERT_EQ(graph.TensorSourceID(node2.spec.Output(0)), 1);
  ASSERT_EQ(graph.TensorIdxInSource(node2.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node2.spec.Output(0)));
  ASSERT_EQ(graph.Tensor(1).name, "copy_data_cpu");
  ASSERT_EQ(graph.Tensor(1).producer.node, 1);
  ASSERT_EQ(graph.Tensor(1).consumers.size(), 0);
  ASSERT_EQ(node2.parent_tensors.size(), 1);
  ASSERT_EQ(node2.parent_tensors[0], 0);
  ASSERT_EQ(node2.children_tensors.size(), 1);
  ASSERT_EQ(node2.children_tensors[0], 1);

  vector<TensorMeta> meta = graph.TensorConsumerMeta(node2.spec.Input(0));
  ASSERT_EQ(meta.size(), 1);
  ASSERT_EQ(meta[0].node, 1);
  ASSERT_EQ(meta[0].index, 0);
  ASSERT_EQ(meta[0].storage_device, StorageDevice::CPU);
}

TEST_F(OpGraphTest, TestGPUOnly) {
  OpGraph graph;

  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "gpu")
          .AddOutput("external_data", StorageDevice::GPU)), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("Copy")
          .AddArg("device", "gpu")
          .AddInput("external_data", StorageDevice::GPU)
          .AddOutput("copy_data", StorageDevice::GPU)), "");

  // Validate the graph
  ASSERT_EQ(graph.NumOp(OpType::CPU), 0);
  ASSERT_EQ(graph.NumOp(OpType::MIXED), 0);
  ASSERT_EQ(graph.NumOp(OpType::GPU), 2);
  ASSERT_EQ(graph.NumTensor(), 2);

  // Validate the source op
  auto& node = graph.Node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 1);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(node.children.count(1), 1);
  ASSERT_EQ(graph.TensorSourceID(node.spec.Output(0)), 0);
  ASSERT_EQ(graph.TensorIdxInSource(node.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<GPUBackend>(node.spec.Output(0)));
  ASSERT_EQ(graph.Tensor(0).name, "external_data_gpu");
  ASSERT_EQ(graph.Tensor(0).producer.node, 0);
  ASSERT_EQ(graph.Tensor(0).consumers.size(), 1);
  ASSERT_EQ(graph.Tensor(0).consumers[0].node, 1);
  ASSERT_EQ(node.parent_tensors.size(), 0);
  ASSERT_EQ(node.children_tensors.size(), 1);
  ASSERT_EQ(node.children_tensors[0], 0);

  // Validate copy op
  auto& node2 = graph.Node(1);
  ASSERT_EQ(node2.id, 1);
  ASSERT_EQ(node2.children.size(), 0);
  ASSERT_EQ(node2.parents.size(), 1);
  ASSERT_EQ(node2.parents.count(0), 1);
  ASSERT_EQ(graph.TensorSourceID(node2.spec.Output(0)), 1);
  ASSERT_EQ(graph.TensorIdxInSource(node2.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<GPUBackend>(node2.spec.Output(0)));
  ASSERT_EQ(graph.Tensor(1).name, "copy_data_gpu");
  ASSERT_EQ(graph.Tensor(1).producer.node, 1);
  ASSERT_EQ(graph.Tensor(1).consumers.size(), 0);
  ASSERT_EQ(node2.parent_tensors.size(), 1);
  ASSERT_EQ(node2.parent_tensors[0], 0);
  ASSERT_EQ(node2.children_tensors.size(), 1);
  ASSERT_EQ(node2.children_tensors[0], 1);

  vector<TensorMeta> meta = graph.TensorConsumerMeta(node2.spec.Input(0));
  ASSERT_EQ(meta.size(), 1);
  ASSERT_EQ(meta[0].node, 1);
  ASSERT_EQ(meta[0].index, 0);
  ASSERT_EQ(meta[0].storage_device, StorageDevice::GPU);
}

TEST_F(OpGraphTest, TestCPUToGPU) {
  OpGraph graph;

  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "cpu")
          .AddOutput("external_data", StorageDevice::CPU)), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("MakeContiguous")
          .AddArg("device", "mixed")
          .AddInput("external_data", StorageDevice::CPU)
          .AddOutput("external_data", StorageDevice::GPU)), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("Copy")
          .AddArg("device", "gpu")
          .AddInput("external_data", StorageDevice::GPU)
          .AddOutput("copy_data", StorageDevice::GPU)), "");

  // Validate the graph
  ASSERT_EQ(graph.NumOp(OpType::CPU), 1);
  ASSERT_EQ(graph.NumOp(OpType::MIXED), 1);
  ASSERT_EQ(graph.NumOp(OpType::GPU), 1);
  ASSERT_EQ(graph.NumTensor(), 3);

  // Validate the source op
  auto& node = graph.Node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 1);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(node.children.count(1), 1);
  ASSERT_EQ(graph.TensorSourceID(node.spec.Output(0)), 0);
  ASSERT_EQ(graph.TensorIdxInSource(node.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node.spec.Output(0)));
  ASSERT_EQ(graph.Tensor(0).name, "external_data_cpu");
  ASSERT_EQ(graph.Tensor(0).producer.node, 0);
  ASSERT_EQ(graph.Tensor(0).consumers.size(), 1);
  ASSERT_EQ(graph.Tensor(0).consumers[0].node, 1);
  ASSERT_EQ(node.parent_tensors.size(), 0);
  ASSERT_EQ(node.children_tensors.size(), 1);
  ASSERT_EQ(node.children_tensors[0], 0);

  // Validate copy-to-dev op
  auto& node2 = graph.Node(1);
  ASSERT_EQ(node2.id, 1);
  ASSERT_EQ(node2.children.size(), 1);
  ASSERT_EQ(node2.parents.size(), 1);
  ASSERT_EQ(node2.parents.count(0), 1);
  ASSERT_EQ(node2.children.count(2), 1);
  ASSERT_EQ(graph.TensorSourceID(node2.spec.Output(0)), 1);
  ASSERT_EQ(graph.TensorIdxInSource(node2.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<GPUBackend>(node2.spec.Output(0)));
  ASSERT_EQ(graph.Tensor(1).name, "external_data_gpu");
  ASSERT_EQ(graph.Tensor(1).producer.node, 1);
  ASSERT_EQ(graph.Tensor(1).consumers.size(), 1);
  ASSERT_EQ(graph.Tensor(1).consumers[0].node, 2);
  ASSERT_EQ(node2.parent_tensors.size(), 1);
  ASSERT_EQ(node2.parent_tensors[0], 0);
  ASSERT_EQ(node2.children_tensors.size(), 1);
  ASSERT_EQ(node2.children_tensors[0], 1);

  vector<TensorMeta> meta = graph.TensorConsumerMeta(node2.spec.Input(0));
  ASSERT_EQ(meta.size(), 1);
  ASSERT_EQ(meta[0].node, 1);
  ASSERT_EQ(meta[0].index, 0);
  ASSERT_EQ(meta[0].storage_device, StorageDevice::CPU);

  // Validate copy op
  auto& node3 = graph.Node(2);
  ASSERT_EQ(node3.id, 2);
  ASSERT_EQ(node3.children.size(), 0);
  ASSERT_EQ(node3.parents.size(), 1);
  ASSERT_EQ(node3.parents.count(1), 1);
  ASSERT_EQ(graph.TensorSourceID(node3.spec.Output(0)), 2);
  ASSERT_EQ(graph.TensorIdxInSource(node3.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<GPUBackend>(node3.spec.Output(0)));
  ASSERT_EQ(graph.Tensor(2).name, "copy_data_gpu");
  ASSERT_EQ(graph.Tensor(2).producer.node, 2);
  ASSERT_EQ(graph.Tensor(2).consumers.size(), 0);
  ASSERT_EQ(node3.parent_tensors.size(), 1);
  ASSERT_EQ(node3.parent_tensors[0], 1);
  ASSERT_EQ(node3.children_tensors.size(), 1);
  ASSERT_EQ(node3.children_tensors[0], 2);


  meta = graph.TensorConsumerMeta(node3.spec.Input(0));
  ASSERT_EQ(meta.size(), 1);
  ASSERT_EQ(meta[0].node, 2);
  ASSERT_EQ(meta[0].index, 0);
  ASSERT_EQ(meta[0].storage_device, StorageDevice::GPU);
}

TEST_F(OpGraphTest, TestGPUThenCPUTopological) {
  OpGraph graph;

  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "gpu")
          .AddOutput("external_dev_data", StorageDevice::GPU)), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("Copy")
          .AddArg("device", "gpu")
          .AddInput("external_dev_data", StorageDevice::GPU)
          .AddOutput("copy_data", StorageDevice::GPU)), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "cpu")
          .AddOutput("external_host_data", StorageDevice::CPU)), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("Copy")
          .AddArg("device", "cpu")
          .AddInput("external_host_data", StorageDevice::CPU)
          .AddOutput("copy_data", StorageDevice::CPU)), "");

  // Validate the graph
  ASSERT_EQ(graph.NumOp(OpType::CPU), 2);
  ASSERT_EQ(graph.NumOp(OpType::MIXED), 0);
  ASSERT_EQ(graph.NumOp(OpType::GPU), 2);
  ASSERT_EQ(graph.NumTensor(), 4);

  // Validate the gpu source op
  auto& node = graph.Node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 1);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(node.children.count(1), 1);
  ASSERT_EQ(graph.TensorSourceID(node.spec.Output(0)), 0);
  ASSERT_EQ(graph.TensorIdxInSource(node.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<GPUBackend>(node.spec.Output(0)));
  ASSERT_EQ(graph.Tensor(0).name, "external_dev_data_gpu");
  ASSERT_EQ(graph.Tensor(0).producer.node, 0);
  ASSERT_EQ(graph.Tensor(0).consumers.size(), 1);
  ASSERT_EQ(graph.Tensor(0).consumers[0].node, 1);
  ASSERT_EQ(node.parent_tensors.size(), 0);
  ASSERT_EQ(node.children_tensors.size(), 1);
  ASSERT_EQ(node.children_tensors[0], 0);

  // Validate gpu copy op
  auto& node2 = graph.Node(1);
  ASSERT_EQ(node2.id, 1);
  ASSERT_EQ(node2.children.size(), 0);
  ASSERT_EQ(node2.parents.size(), 1);
  ASSERT_EQ(node2.parents.count(0), 1);
  ASSERT_EQ(graph.TensorSourceID(node2.spec.Output(0)), 1);
  ASSERT_EQ(graph.TensorIdxInSource(node2.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<GPUBackend>(node2.spec.Output(0)));
  ASSERT_EQ(graph.Tensor(1).name, "copy_data_gpu");
  ASSERT_EQ(graph.Tensor(1).producer.node, 1);
  ASSERT_EQ(graph.Tensor(1).consumers.size(), 0);
  ASSERT_EQ(node2.parent_tensors.size(), 1);
  ASSERT_EQ(node2.parent_tensors[0], 0);
  ASSERT_EQ(node2.children_tensors.size(), 1);
  ASSERT_EQ(node2.children_tensors[0], 1);

  vector<TensorMeta> meta = graph.TensorConsumerMeta(node2.spec.Input(0));
  ASSERT_EQ(meta.size(), 1);
  ASSERT_EQ(meta[0].node, 1);
  ASSERT_EQ(meta[0].index, 0);
  ASSERT_EQ(meta[0].storage_device, StorageDevice::GPU);

  // Validate cpu source op
  auto& node3 = graph.Node(2);
  ASSERT_EQ(node3.id, 2);
  ASSERT_EQ(node3.children.size(), 1);
  ASSERT_EQ(node3.parents.size(), 0);
  ASSERT_EQ(node3.children.count(3), 1);
  ASSERT_EQ(graph.TensorSourceID(node3.spec.Output(0)), 2);
  ASSERT_EQ(graph.TensorIdxInSource(node3.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node3.spec.Output(0)));
  ASSERT_EQ(graph.Tensor(2).name, "external_host_data_cpu");
  ASSERT_EQ(graph.Tensor(2).producer.node, 2);
  ASSERT_EQ(graph.Tensor(2).consumers.size(), 1);
  ASSERT_EQ(graph.Tensor(2).consumers[0].node, 3);
  ASSERT_EQ(node3.parent_tensors.size(), 0);
  ASSERT_EQ(node3.children_tensors.size(), 1);
  ASSERT_EQ(node3.children_tensors[0], 2);

  // Validate cpu copy op
  auto& node4 = graph.Node(3);
  ASSERT_EQ(node4.id, 3);
  ASSERT_EQ(node4.children.size(), 0);
  ASSERT_EQ(node4.parents.size(), 1);
  ASSERT_EQ(node4.parents.count(2), 1);
  ASSERT_EQ(graph.TensorSourceID(node4.spec.Output(0)), 3);
  ASSERT_EQ(graph.TensorIdxInSource(node4.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node4.spec.Output(0)));
  ASSERT_EQ(graph.Tensor(3).name, "copy_data_cpu");
  ASSERT_EQ(graph.Tensor(3).producer.node, 3);
  ASSERT_EQ(graph.Tensor(3).consumers.size(), 0);
  ASSERT_EQ(node4.parent_tensors.size(), 1);
  ASSERT_EQ(node4.parent_tensors[0], 2);
  ASSERT_EQ(node4.children_tensors.size(), 1);
  ASSERT_EQ(node4.children_tensors[0], 3);

  meta = graph.TensorConsumerMeta(node4.spec.Input(0));
  ASSERT_EQ(meta.size(), 1);
  ASSERT_EQ(meta[0].node, 3);
  ASSERT_EQ(meta[0].index, 0);
  ASSERT_EQ(meta[0].storage_device, StorageDevice::CPU);
}

TEST_F(OpGraphTest, TestOpRemoval) {
  OpGraph graph;

  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddOutput("data_1", StorageDevice::CPU)
          .AddOutput("data_2", StorageDevice::CPU)), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddArg("num_outputs", 1)
          .AddInput("data_2", StorageDevice::CPU)
          .AddInput("data_1", StorageDevice::CPU)
          .AddOutput("dummy_out", StorageDevice::CPU)), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddArg("num_outputs", 1)
          .AddInput("data_1", StorageDevice::CPU)
          .AddOutput("dummy_out_two", StorageDevice::CPU)), "");

  // Validate the graph
  ASSERT_EQ(graph.NumOp(OpType::CPU), 3);
  ASSERT_EQ(graph.NumOp(OpType::MIXED), 0);
  ASSERT_EQ(graph.NumOp(OpType::GPU), 0);
  ASSERT_EQ(graph.NumTensor(), 4);

  // Validate the dummy source op
  auto& node = graph.Node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 2);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(node.children.count(1), 1);
  ASSERT_EQ(node.children.count(2), 1);
  ASSERT_EQ(graph.TensorSourceID(node.spec.Output(0)), 0);
  ASSERT_EQ(graph.TensorIdxInSource(node.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node.spec.Output(0)));
  ASSERT_EQ(graph.Tensor(0).name, "data_1_cpu");
  ASSERT_EQ(graph.Tensor(0).producer.node, 0);
  ASSERT_EQ(graph.Tensor(0).consumers.size(), 2);
  std::vector<OpNodeId> cons = {graph.Tensor(0).consumers[0].node,
                                graph.Tensor(0).consumers[1].node};
  std::sort(cons.begin(), cons.end());
  auto expected_cons = std::vector<OpNodeId>{1, 2};
  ASSERT_EQ(cons, expected_cons);
  ASSERT_EQ(graph.Tensor(1).name, "data_2_cpu");
  ASSERT_EQ(graph.Tensor(1).producer.node, 0);
  ASSERT_EQ(graph.Tensor(1).consumers.size(), 1);
  ASSERT_EQ(graph.Tensor(1).consumers[0].node, 1);
  ASSERT_EQ(node.parent_tensors.size(), 0);
  ASSERT_EQ(node.children_tensors.size(), 2);
  ASSERT_EQ(node.children_tensors[0], 0);
  ASSERT_EQ(node.children_tensors[1], 1);

  // Validate dummy op 1
  auto& node2 = graph.Node(1);
  ASSERT_EQ(node2.id, 1);
  ASSERT_EQ(node2.children.size(), 0);
  ASSERT_EQ(node2.parents.size(), 1);
  ASSERT_EQ(node2.parents.count(0), 1);
  ASSERT_EQ(graph.TensorSourceID(node2.spec.Output(0)), 1);
  ASSERT_EQ(graph.TensorIdxInSource(node2.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node2.spec.Output(0)));
  ASSERT_EQ(graph.Tensor(2).name, "dummy_out_cpu");
  ASSERT_EQ(graph.Tensor(2).producer.node, 1);
  ASSERT_EQ(graph.Tensor(2).consumers.size(), 0);
  ASSERT_EQ(node2.parent_tensors.size(), 2);
  ASSERT_EQ(node2.parent_tensors[0], 1);
  ASSERT_EQ(node2.parent_tensors[1], 0);
  ASSERT_EQ(node2.children_tensors.size(), 1);
  ASSERT_EQ(node2.children_tensors[0], 2);

  vector<TensorMeta> meta = graph.TensorConsumerMeta(node2.spec.Input(0));
  ASSERT_EQ(meta.size(), 1);
  ASSERT_EQ(meta[0].node, 1);
  ASSERT_EQ(meta[0].index, 0);
  ASSERT_EQ(meta[0].storage_device, StorageDevice::CPU);

  // Validate dummy op 2
  auto& node3 = graph.Node(2);
  ASSERT_EQ(node3.id, 2);
  ASSERT_EQ(node3.children.size(), 0);
  ASSERT_EQ(node3.parents.size(), 1);
  ASSERT_EQ(node3.parents.count(0), 1);
  ASSERT_EQ(graph.TensorSourceID(node3.spec.Output(0)), 2);
  ASSERT_EQ(graph.TensorIdxInSource(node3.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node3.spec.Output(0)));
  ASSERT_EQ(graph.Tensor(3).name, "dummy_out_two_cpu");
  ASSERT_EQ(graph.Tensor(3).producer.node, 2);
  ASSERT_EQ(graph.Tensor(3).consumers.size(), 0);
  ASSERT_EQ(node3.parent_tensors.size(), 1);
  ASSERT_EQ(node3.parent_tensors[0], 0);
  ASSERT_EQ(node3.children_tensors.size(), 1);
  ASSERT_EQ(node3.children_tensors[0], 3);

  // Input zero is also consumed (as input 1) to op 1
  meta = graph.TensorConsumerMeta(node3.spec.Input(0));
  ASSERT_EQ(meta.size(), 2);
  ASSERT_EQ(meta[0].node, 1);
  ASSERT_EQ(meta[0].index, 1);
  ASSERT_EQ(meta[0].storage_device, StorageDevice::CPU);
  ASSERT_EQ(meta[1].node, 2);
  ASSERT_EQ(meta[1].index, 0);
  ASSERT_EQ(meta[1].storage_device, StorageDevice::CPU);

  // Remove op 1
  graph.RemoveOp(1);

  // Validate the updated graph
  ASSERT_EQ(graph.NumOp(OpType::CPU), 2);
  ASSERT_EQ(graph.NumOp(OpType::MIXED), 0);
  ASSERT_EQ(graph.NumOp(OpType::GPU), 0);
  ASSERT_EQ(graph.NumTensor(), 3);

  // Validate the source op
  auto& node4 = graph.Node(0);
  ASSERT_EQ(node4.id, 0);
  ASSERT_EQ(node4.children.size(), 1);
  ASSERT_EQ(node4.parents.size(), 0);
  ASSERT_EQ(node4.children.count(1), 1);
  ASSERT_EQ(graph.TensorSourceID(node4.spec.Output(0)), 0);
  ASSERT_EQ(graph.TensorIdxInSource(node4.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node4.spec.Output(0)));
  ASSERT_EQ(graph.Tensor(0).name, "data_1_cpu");
  ASSERT_EQ(graph.Tensor(0).producer.node, 0);
  ASSERT_EQ(graph.Tensor(0).consumers.size(), 1);
  ASSERT_EQ(graph.Tensor(0).consumers[0].node, 1);
  ASSERT_EQ(graph.Tensor(1).name, "data_2_cpu");
  ASSERT_EQ(graph.Tensor(1).producer.node, 0);
  ASSERT_EQ(graph.Tensor(1).consumers.size(), 0);
  ASSERT_EQ(node4.parent_tensors.size(), 0);
  ASSERT_EQ(node4.children_tensors.size(), 2);
  ASSERT_EQ(node4.children_tensors[0], 0);
  ASSERT_EQ(node4.children_tensors[1], 1);

  // Validate copy op 1
  auto& node5 = graph.Node(1);
  ASSERT_EQ(node5.id, 1);
  ASSERT_EQ(node5.children.size(), 0);
  ASSERT_EQ(node5.parents.size(), 1);
  ASSERT_EQ(node5.parents.count(0), 1);
  ASSERT_EQ(node5.spec.NumInput(), 1);
  ASSERT_EQ(node5.spec.NumOutput(), 1);
  ASSERT_EQ(graph.TensorSourceID(node5.spec.Output(0)), 1);
  ASSERT_EQ(graph.TensorIdxInSource(node5.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node5.spec.Output(0)));
  ASSERT_EQ(graph.Tensor(2).name, "dummy_out_two_cpu");
  ASSERT_EQ(graph.Tensor(2).producer.node, 1);
  ASSERT_EQ(graph.Tensor(2).consumers.size(), 0);
  ASSERT_EQ(node5.parent_tensors.size(), 1);
  ASSERT_EQ(node5.parent_tensors[0], 0);
  ASSERT_EQ(node5.children_tensors.size(), 1);
  ASSERT_EQ(node5.children_tensors[0], 2);

  meta = graph.TensorConsumerMeta(node5.spec.Input(0));
  ASSERT_EQ(meta.size(), 1);
  ASSERT_EQ(meta[0].node, 1);
  ASSERT_EQ(meta[0].index, 0);
  ASSERT_EQ(meta[0].storage_device, StorageDevice::CPU);
}

TEST_F(OpGraphTest, TestFailureCPUOpGPUInput) {
  OpGraph graph;

  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "gpu")
          .AddOutput("external_data", StorageDevice::GPU)), "");

  ASSERT_THROW(
      graph.AddOp(this->PrepareSpec(
              OpSpec("Copy")
              .AddArg("device", "cpu")
              .AddInput("external_data", StorageDevice::GPU)
              .AddOutput("copy_data", StorageDevice::CPU)), ""),
      std::runtime_error);
}

TEST_F(OpGraphTest, TestFailureCPUToGPUOp) {
  OpGraph graph;

  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "gpu")
          .AddOutput("external_data", StorageDevice::GPU)), "");

  ASSERT_THROW(
      graph.AddOp(this->PrepareSpec(
              OpSpec("Copy")
              .AddArg("device", "cpu")
              .AddInput("external_data", StorageDevice::CPU)
              .AddOutput("copy_data", StorageDevice::CPU)), ""),
      std::runtime_error);
}

TEST_F(OpGraphTest, TestFailureNonTopological) {
  OpGraph graph;

  ASSERT_THROW(
      graph.AddOp(this->PrepareSpec(
              OpSpec("Copy")
              .AddArg("device", "cpu")
              .AddInput("external_data", StorageDevice::CPU)
              .AddOutput("copy_data", StorageDevice::CPU)), ""),
      std::runtime_error);

  // Note: Just to make it clear what this verifies...
  // graph.AddOp(this->PrepareSpec(
  //         OpSpec("ExternalSource")
  //         .AddArg("device", "cpu")
  //         .AddOutput("external_data", StorageDevice::CPU)
  //         ), "");
}

TEST_F(OpGraphTest, TestFailureCircularOp) {
  OpGraph graph;

  ASSERT_THROW(
      graph.AddOp(this->PrepareSpec(
              OpSpec("Copy")
              .AddArg("device", "cpu")
              .AddInput("data", StorageDevice::CPU)
              .AddOutput("data", StorageDevice::CPU)), ""),
      std::runtime_error);
}

TEST_F(OpGraphTest, TestGetTensorOrigin) {
  OpGraph graph;

  // The nodes are numbered in the order of addition, top to bottom in graph.
  graph.AddOp(this->PrepareSpec(OpSpec("ExternalSource")
                                    .AddArg("device", "cpu")
                                    .AddArg("device_id", 0)
                                    .AddOutput("data", StorageDevice::CPU)),
              "ExternalSource");  // tensor node 0

  graph.AddOp(this->PrepareSpec(OpSpec("Copy")
                                    .AddInput("data", StorageDevice::CPU)
                                    .AddOutput("copy_0_data", StorageDevice::CPU)),
              "Copy0");  // tensor node 1

  graph.AddOp(this->PrepareSpec(OpSpec("MakeContiguous")
                                    .AddInput("copy_0_data", StorageDevice::CPU)
                                    .AddOutput("contiguous_data", StorageDevice::CPU)),
              "MakeContiguous");  // tensor node 2

  graph.AddOp(this->PrepareSpec(OpSpec("PassthroughOp")
                                    .AddInput("contiguous_data", StorageDevice::CPU)
                                    .AddOutput("passthrough_data", StorageDevice::CPU)),
              "Passthrough");  // tensor node 3


  graph.AddOp(this->PrepareSpec(OpSpec("Copy")
                                    .AddInput("passthrough_data", StorageDevice::CPU)
                                    .AddOutput("copy_1_data", StorageDevice::CPU)),
              "Copy1");  // tensor node 4

  graph.InstantiateOperators();

  // we didn't compute pass through for MakeContiguous
  EXPECT_THROW(graph.GetTensorOrigin(0), std::runtime_error);

  graph.SetupMakeContiguousPassThrough();

  // Entry point to the graph
  auto origin_0 = std::vector<TensorNodeId>{0};
  EXPECT_EQ(graph.GetTensorOrigin(0), origin_0);
  // Copy doesn't pass through
  auto origin_1 = std::vector<TensorNodeId>{1};
  EXPECT_EQ(graph.GetTensorOrigin(1), origin_1);
  // Make Contiguous passes through a contiguous output from copy
  auto origin_2 = std::vector<TensorNodeId>{2, 1};
  EXPECT_EQ(graph.GetTensorOrigin(2), origin_2);
  // Same as above, and Reshape is always Pass Through
  auto origin_3 = std::vector<TensorNodeId>{3, 2, 1};
  EXPECT_EQ(graph.GetTensorOrigin(3), origin_3);
  // Copy doesn't pass through
  auto origin_4 = std::vector<TensorNodeId>{4};
  EXPECT_EQ(graph.GetTensorOrigin(4), origin_4);
}

inline bool operator==(const dali::TensorMeta &a, const dali::TensorMeta &b) {
  return a.index == b.index && a.node == b.node && a.storage_device == b.storage_device;
}

void CheckEqual(const OpGraph &g1, const OpGraph &g2) {
  EXPECT_EQ(g1.NumOp(), g2.NumOp()) << "The number of operator nodes differs.";
  EXPECT_EQ(g1.NumTensor(), g2.NumTensor()) << "The number of tensor nodes differs.";
  EXPECT_EQ(g1.NumOp(OpType::CPU), g2.NumOp(OpType::CPU)) << "The numberof CPU nodes differs.";
  EXPECT_EQ(g1.NumOp(OpType::GPU), g2.NumOp(OpType::GPU)) << "The numberof GPU nodes differs.";
  EXPECT_EQ(g1.NumOp(OpType::MIXED), g2.NumOp(OpType::MIXED))
        << "The numberof mixed nodes differs.";

  if (::testing::Test::HasFailure())
    return;

  for (int i = 0; i < g1.NumOp(); i++) {
    auto &n1 = g1.Node(i);
    auto &n2 = g2.Node(i);
    EXPECT_EQ(n1.id, n2.id) << " @ node " << i;
    EXPECT_EQ(n1.instance_name, n2.instance_name) << " @ node " << i;
    EXPECT_EQ(n1.spec.SchemaName(), n2.spec.SchemaName())<< " @ node " << i;
    EXPECT_EQ(n1.children, n2.children) << " @ node " << i;
    EXPECT_EQ(n1.parents, n2.parents) << " @ node " << i;
  }
  for (int i = 0; i < g1.NumTensor(); i++) {
    auto &t1 = g1.Tensor(i);
    auto &t2 = g2.Tensor(i);
    EXPECT_EQ(t1.id, t2.id) << " @ node " << i;
    EXPECT_EQ(t1.name, t2.name) << " @ node " << i;
    EXPECT_EQ(t1.consumers, t2.consumers) << " @ node " << i;
    EXPECT_EQ(t1.producer, t2.producer) << " @ node " << i;
  }
}

TEST_F(OpGraphTest, Lowering) {
  OpSpec spec0 = this->PrepareSpec(OpSpec("ExternalSource")
    .AddArg("device", "cpu")
    .AddArg("device_id", 0)
    .AddOutput("data", StorageDevice::CPU));

  OpSpec spec1 = this->PrepareSpec(OpSpec("Copy")
    .AddInput("data", StorageDevice::CPU)
    .AddOutput("copy_0_data", StorageDevice::CPU));

  OpSpec spec2 = this->PrepareSpec(OpSpec("MakeContiguous")
    .AddInput("copy_0_data", StorageDevice::CPU)
    .AddOutput("contiguous_data", StorageDevice::CPU));

  OpSpec spec3 = this->PrepareSpec(OpSpec("PassthroughOp")
    .AddInput("contiguous_data", StorageDevice::CPU)
    .AddOutput("passthrough_data", StorageDevice::CPU));

  OpSpec spec4 = this->PrepareSpec(OpSpec("Copy")
    .AddInput("passthrough_data", StorageDevice::CPU)
    .AddOutput("copy_1_data", StorageDevice::CPU));

  graph::OpGraph::Builder b;
  // This is the same graph as in TestGetTensorOrigin, but the topological order is not maintained.
  b.Add("Copy1", spec4);  // tensor node 4
  b.Add("ExternalSource", spec0);  // tensor node 0
  b.Add("MakeContiguous", spec2);  // tensor node 2
  b.Add("Passthrough", spec3);  // tensor node 3
  b.Add("Copy0", spec1);  // tensor node 1
  b.AddOutput("copy_1_data_cpu");

  auto def = std::move(b).GetGraph(true);
  OpGraph lowered;
  lowered.Lower(def);

  OpGraph handmade;
  handmade.AddOp(spec0, "ExternalSource");  // tensor node 0
  handmade.AddOp(spec1, "Copy0");  // tensor node 1
  handmade.AddOp(spec2, "MakeContiguous");  // tensor node 2
  handmade.AddOp(spec3, "Passthrough");  // tensor node 3
  handmade.AddOp(spec4, "Copy1");

  CheckEqual(lowered, handmade);
}


}  // namespace dali
