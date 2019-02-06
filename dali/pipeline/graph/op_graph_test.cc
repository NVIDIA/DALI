// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/graph/op_graph.h"

#include <gtest/gtest.h>

#include "dali/test/dali_test.h"

namespace dali {

class OpGraphTest : public DALITest {
 public:
  inline OpSpec PrepareSpec(OpSpec spec) {
    spec.AddArg("batch_size", 1)
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
          .AddOutput("external_data", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("Copy")
          .AddInput("external_data", "cpu")
          .AddOutput("copy_data", "cpu")), "");

  // Validate the graph
  ASSERT_EQ(graph.NumOp(DALIOpType::CPU), 2);
  ASSERT_EQ(graph.NumOp(DALIOpType::MIXED), 0);
  ASSERT_EQ(graph.NumOp(DALIOpType::GPU), 0);

  // Validate the source op
  auto& node = graph.Node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 1);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(node.children.count(1), 1);
  ASSERT_EQ(graph.TensorSourceID(node.spec.Output(0)), 0);
  ASSERT_EQ(graph.TensorIdxInSource(node.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node.spec.Output(0)));

  // Validate copy op
  auto& node2 = graph.Node(1);
  ASSERT_EQ(node2.id, 1);
  ASSERT_EQ(node2.children.size(), 0);
  ASSERT_EQ(node2.parents.size(), 1);
  ASSERT_EQ(node2.parents.count(0), 1);
  ASSERT_EQ(graph.TensorSourceID(node2.spec.Output(0)), 1);
  ASSERT_EQ(graph.TensorIdxInSource(node2.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node2.spec.Output(0)));

  vector<TensorMeta> meta = graph.TensorConsumerMeta(node2.spec.Input(0));
  ASSERT_EQ(meta.size(), 1);
  ASSERT_EQ(meta[0].node, 1);
  ASSERT_EQ(meta[0].index, 0);
  ASSERT_EQ(meta[0].storage_device, DALITensorDevice::CPU);
}

TEST_F(OpGraphTest, TestGPUOnly) {
  OpGraph graph;

  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "gpu")
          .AddOutput("external_data", "gpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("Copy")
          .AddArg("device", "gpu")
          .AddInput("external_data", "gpu")
          .AddOutput("copy_data", "gpu")), "");

  // Validate the graph
  ASSERT_EQ(graph.NumOp(DALIOpType::CPU), 0);
  ASSERT_EQ(graph.NumOp(DALIOpType::MIXED), 0);
  ASSERT_EQ(graph.NumOp(DALIOpType::GPU), 2);

  // Validate the source op
  auto& node = graph.Node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 1);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(node.children.count(1), 1);
  ASSERT_EQ(graph.TensorSourceID(node.spec.Output(0)), 0);
  ASSERT_EQ(graph.TensorIdxInSource(node.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<GPUBackend>(node.spec.Output(0)));

  // Validate copy op
  auto& node2 = graph.Node(1);
  ASSERT_EQ(node2.id, 1);
  ASSERT_EQ(node2.children.size(), 0);
  ASSERT_EQ(node2.parents.size(), 1);
  ASSERT_EQ(node2.parents.count(0), 1);
  ASSERT_EQ(graph.TensorSourceID(node2.spec.Output(0)), 1);
  ASSERT_EQ(graph.TensorIdxInSource(node2.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<GPUBackend>(node2.spec.Output(0)));

  vector<TensorMeta> meta = graph.TensorConsumerMeta(node2.spec.Input(0));
  ASSERT_EQ(meta.size(), 1);
  ASSERT_EQ(meta[0].node, 1);
  ASSERT_EQ(meta[0].index, 0);
  ASSERT_EQ(meta[0].storage_device, DALITensorDevice::GPU);
}

TEST_F(OpGraphTest, TestCPUToGPU) {
  OpGraph graph;

  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "cpu")
          .AddOutput("external_data", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("MakeContiguous")
          .AddArg("device", "mixed")
          .AddInput("external_data", "cpu")
          .AddOutput("external_data", "gpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("Copy")
          .AddArg("device", "gpu")
          .AddInput("external_data", "gpu")
          .AddOutput("copy_data", "gpu")), "");

  // Validate the graph
  ASSERT_EQ(graph.NumOp(DALIOpType::CPU), 1);
  ASSERT_EQ(graph.NumOp(DALIOpType::MIXED), 1);
  ASSERT_EQ(graph.NumOp(DALIOpType::GPU), 1);

  // Validate the source op
  auto& node = graph.Node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 1);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(node.children.count(1), 1);
  ASSERT_EQ(graph.TensorSourceID(node.spec.Output(0)), 0);
  ASSERT_EQ(graph.TensorIdxInSource(node.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node.spec.Output(0)));

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

  vector<TensorMeta> meta = graph.TensorConsumerMeta(node2.spec.Input(0));
  ASSERT_EQ(meta.size(), 1);
  ASSERT_EQ(meta[0].node, 1);
  ASSERT_EQ(meta[0].index, 0);
  ASSERT_EQ(meta[0].storage_device, DALITensorDevice::CPU);

  // Validate copy op
  auto& node3 = graph.Node(2);
  ASSERT_EQ(node3.id, 2);
  ASSERT_EQ(node3.children.size(), 0);
  ASSERT_EQ(node3.parents.size(), 1);
  ASSERT_EQ(node3.parents.count(1), 1);
  ASSERT_EQ(graph.TensorSourceID(node3.spec.Output(0)), 2);
  ASSERT_EQ(graph.TensorIdxInSource(node3.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<GPUBackend>(node3.spec.Output(0)));

  meta = graph.TensorConsumerMeta(node3.spec.Input(0));
  ASSERT_EQ(meta.size(), 1);
  ASSERT_EQ(meta[0].node, 2);
  ASSERT_EQ(meta[0].index, 0);
  ASSERT_EQ(meta[0].storage_device, DALITensorDevice::GPU);
}

TEST_F(OpGraphTest, TestGPUThenCPUTopological) {
  OpGraph graph;

  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "gpu")
          .AddOutput("external_dev_data", "gpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("Copy")
          .AddArg("device", "gpu")
          .AddInput("external_dev_data", "gpu")
          .AddOutput("copy_data", "gpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "cpu")
          .AddOutput("external_host_data", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("Copy")
          .AddArg("device", "cpu")
          .AddInput("external_host_data", "cpu")
          .AddOutput("copy_data", "cpu")), "");

  // Validate the graph
  ASSERT_EQ(graph.NumOp(DALIOpType::CPU), 2);
  ASSERT_EQ(graph.NumOp(DALIOpType::MIXED), 0);
  ASSERT_EQ(graph.NumOp(DALIOpType::GPU), 2);

  // Validate the gpu source op
  auto& node = graph.Node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 1);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(node.children.count(1), 1);
  ASSERT_EQ(graph.TensorSourceID(node.spec.Output(0)), 0);
  ASSERT_EQ(graph.TensorIdxInSource(node.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<GPUBackend>(node.spec.Output(0)));

  // Validate gpu copy op
  auto& node2 = graph.Node(1);
  ASSERT_EQ(node2.id, 1);
  ASSERT_EQ(node2.children.size(), 0);
  ASSERT_EQ(node2.parents.size(), 1);
  ASSERT_EQ(node2.parents.count(0), 1);
  ASSERT_EQ(graph.TensorSourceID(node2.spec.Output(0)), 1);
  ASSERT_EQ(graph.TensorIdxInSource(node2.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<GPUBackend>(node2.spec.Output(0)));

  vector<TensorMeta> meta = graph.TensorConsumerMeta(node2.spec.Input(0));
  ASSERT_EQ(meta.size(), 1);
  ASSERT_EQ(meta[0].node, 1);
  ASSERT_EQ(meta[0].index, 0);
  ASSERT_EQ(meta[0].storage_device, DALITensorDevice::GPU);

  // Validate cpu source op
  auto& node3 = graph.Node(2);
  ASSERT_EQ(node3.id, 2);
  ASSERT_EQ(node3.children.size(), 1);
  ASSERT_EQ(node3.parents.size(), 0);
  ASSERT_EQ(node3.children.count(3), 1);
  ASSERT_EQ(graph.TensorSourceID(node3.spec.Output(0)), 2);
  ASSERT_EQ(graph.TensorIdxInSource(node3.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node3.spec.Output(0)));

  // Validate cpu copy op
  auto& node4 = graph.Node(3);
  ASSERT_EQ(node4.id, 3);
  ASSERT_EQ(node4.children.size(), 0);
  ASSERT_EQ(node4.parents.size(), 1);
  ASSERT_EQ(node4.parents.count(2), 1);
  ASSERT_EQ(graph.TensorSourceID(node4.spec.Output(0)), 3);
  ASSERT_EQ(graph.TensorIdxInSource(node4.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node4.spec.Output(0)));

  meta = graph.TensorConsumerMeta(node4.spec.Input(0));
  ASSERT_EQ(meta.size(), 1);
  ASSERT_EQ(meta[0].node, 3);
  ASSERT_EQ(meta[0].index, 0);
  ASSERT_EQ(meta[0].storage_device, DALITensorDevice::CPU);
}

TEST_F(OpGraphTest, TestOpRemoval) {
  OpGraph graph;

  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddOutput("data_1", "cpu")
          .AddOutput("data_2", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddArg("num_outputs", 1)
          .AddInput("data_2", "cpu")
          .AddInput("data_1", "cpu")
          .AddOutput("dummy_out", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddArg("num_outputs", 1)
          .AddInput("data_1", "cpu")
          .AddOutput("dummy_out_two", "cpu")), "");


  // Validate the graph
  ASSERT_EQ(graph.NumOp(DALIOpType::CPU), 3);
  ASSERT_EQ(graph.NumOp(DALIOpType::MIXED), 0);
  ASSERT_EQ(graph.NumOp(DALIOpType::GPU), 0);

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

  // Validate dummy op 1
  auto& node2 = graph.Node(1);
  ASSERT_EQ(node2.id, 1);
  ASSERT_EQ(node2.children.size(), 0);
  ASSERT_EQ(node2.parents.size(), 1);
  ASSERT_EQ(node2.parents.count(0), 1);
  ASSERT_EQ(graph.TensorSourceID(node2.spec.Output(0)), 1);
  ASSERT_EQ(graph.TensorIdxInSource(node2.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node2.spec.Output(0)));

  vector<TensorMeta> meta = graph.TensorConsumerMeta(node2.spec.Input(0));
  ASSERT_EQ(meta.size(), 1);
  ASSERT_EQ(meta[0].node, 1);
  ASSERT_EQ(meta[0].index, 0);
  ASSERT_EQ(meta[0].storage_device, DALITensorDevice::CPU);

  // Validate dummy op 2
  auto& node3 = graph.Node(2);
  ASSERT_EQ(node3.id, 2);
  ASSERT_EQ(node3.children.size(), 0);
  ASSERT_EQ(node3.parents.size(), 1);
  ASSERT_EQ(node3.parents.count(0), 1);
  ASSERT_EQ(graph.TensorSourceID(node3.spec.Output(0)), 2);
  ASSERT_EQ(graph.TensorIdxInSource(node3.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node3.spec.Output(0)));

  // Input zero is also consumed (as input 1) to op 1
  meta = graph.TensorConsumerMeta(node3.spec.Input(0));
  ASSERT_EQ(meta.size(), 2);
  ASSERT_EQ(meta[0].node, 1);
  ASSERT_EQ(meta[0].index, 1);
  ASSERT_EQ(meta[0].storage_device, DALITensorDevice::CPU);
  ASSERT_EQ(meta[1].node, 2);
  ASSERT_EQ(meta[1].index, 0);
  ASSERT_EQ(meta[1].storage_device, DALITensorDevice::CPU);

  // Remove op 1
  graph.RemoveOp(1);

  // Validate the updated graph
  ASSERT_EQ(graph.NumOp(DALIOpType::CPU), 2);
  ASSERT_EQ(graph.NumOp(DALIOpType::MIXED), 0);
  ASSERT_EQ(graph.NumOp(DALIOpType::GPU), 0);

  // Validate the source op
  auto& node4 = graph.Node(0);
  ASSERT_EQ(node4.id, 0);
  ASSERT_EQ(node4.children.size(), 1);
  ASSERT_EQ(node4.parents.size(), 0);
  ASSERT_EQ(node4.children.count(1), 1);
  ASSERT_EQ(graph.TensorSourceID(node4.spec.Output(0)), 0);
  ASSERT_EQ(graph.TensorIdxInSource(node4.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node4.spec.Output(0)));

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

  meta = graph.TensorConsumerMeta(node5.spec.Input(0));
  ASSERT_EQ(meta.size(), 1);
  ASSERT_EQ(meta[0].node, 1);
  ASSERT_EQ(meta[0].index, 0);
  ASSERT_EQ(meta[0].storage_device, DALITensorDevice::CPU);
}

TEST_F(OpGraphTest, TestFailureCPUOpGPUInput) {
  OpGraph graph;

  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "gpu")
          .AddOutput("external_data", "gpu")), "");

  ASSERT_THROW(
      graph.AddOp(this->PrepareSpec(
              OpSpec("Copy")
              .AddArg("device", "cpu")
              .AddInput("external_data", "gpu")
              .AddOutput("copy_data", "cpu")), ""),
      std::runtime_error);
}

TEST_F(OpGraphTest, TestFailureCPUToGPUOp) {
  OpGraph graph;

  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "gpu")
          .AddOutput("external_data", "gpu")), "");

  ASSERT_THROW(
      graph.AddOp(this->PrepareSpec(
              OpSpec("Copy")
              .AddArg("device", "cpu")
              .AddInput("external_data", "cpu")
              .AddOutput("copy_data", "cpu")), ""),
      std::runtime_error);
}

TEST_F(OpGraphTest, TestFailureNonTopological) {
  OpGraph graph;

  ASSERT_THROW(
      graph.AddOp(this->PrepareSpec(
              OpSpec("Copy")
              .AddArg("device", "cpu")
              .AddInput("external_data", "cpu")
              .AddOutput("copy_data", "cpu")), ""),
      std::runtime_error);

  // Note: Just to make it clear what this verifies...
  // graph.AddOp(this->PrepareSpec(
  //         OpSpec("ExternalSource")
  //         .AddArg("device", "cpu")
  //         .AddOutput("external_data", "cpu")
  //         ), "");
}

TEST_F(OpGraphTest, TestFailureCircularOp) {
  OpGraph graph;

  ASSERT_THROW(
      graph.AddOp(this->PrepareSpec(
              OpSpec("Copy")
              .AddArg("device", "cpu")
              .AddInput("data", "cpu")
              .AddOutput("data", "cpu")), ""),
      std::runtime_error);
}

}  // namespace dali
