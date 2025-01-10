// Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/operator/checkpointing/checkpoint.h"

#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

#include "dali/test/dali_test.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/executor/executor_factory.h"
// TODO(michalz): Use new graph, when ready
#include "dali/pipeline/executor/lowered_graph.h"

namespace dali {

namespace {

void BuildFromLegacyGraph(Checkpoint &checkpoint, const OpGraph &graph) {
  checkpoint.Clear();
  for (const auto &node : graph.GetOpNodes())
    checkpoint.AddOperator(node.instance_name);
}

auto GetSimpleExecutor() {
  return GetExecutor(false, false, false, false, 1, 1, CPU_ONLY_DEVICE_ID, 0);
}

}  // namespace

template <typename Backend>
class DummyOperatorWithState : public Operator<Backend> {};

template<>
class DummyOperatorWithState<CPUBackend> : public Operator<CPUBackend> {
 public:
  explicit DummyOperatorWithState(const OpSpec &spec)
      : Operator<CPUBackend>(spec)
      , state_(spec.GetArgument<uint32_t>("dummy_state")) {}

  void SaveState(OpCheckpoint &cpt, AccessOrder order) override {
    cpt.MutableCheckpointState() = state_;
  }

  void RestoreState(const OpCheckpoint &cpt) override {
    state_ = cpt.CheckpointState<uint32_t>();
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc,
                 const Workspace &ws) override { return false; }

  uint32_t GetState() const { return state_; }

 private:
  uint32_t state_;
};

struct DummyGPUData {
  uint32_t *ptr;
  AccessOrder order;

  inline DummyGPUData(cudaStream_t stream, uint32_t value) : order(stream) {
    CUDA_CALL(cudaMalloc(&ptr, sizeof(uint32_t)));
    CUDA_CALL(cudaMemcpyAsync(ptr, &value, sizeof(uint32_t),
                              cudaMemcpyHostToDevice, stream));
  }

  inline ~DummyGPUData() {
    CUDA_DTOR_CALL(cudaFree(ptr));
  }
};

template<>
class DummyOperatorWithState<GPUBackend> : public Operator<GPUBackend> {
 public:
  explicit DummyOperatorWithState(const OpSpec &spec)
      : Operator<GPUBackend>(spec)
      , state_(spec.GetArgument<cudaStream_t>("cuda_stream"),
               spec.GetArgument<uint32_t>("dummy_state")) {}

  void SaveState(OpCheckpoint &cpt, AccessOrder order) override {
    if (!order.is_device())
      FAIL() << "Cuda stream was not provided for GPU operator checkpointing.";

    std::any &cpt_state = cpt.MutableCheckpointState();
    if (!cpt_state.has_value())
      cpt_state = static_cast<uint32_t>(0);

    cpt.SetOrder(order);
    CUDA_CALL(cudaMemcpyAsync(&std::any_cast<uint32_t &>(cpt_state), state_.ptr,
                              sizeof(uint32_t), cudaMemcpyDeviceToHost, order.stream()));
  }

  void RestoreState(const OpCheckpoint &cpt) override {
    CUDA_CALL(cudaMemcpy(state_.ptr, &cpt.CheckpointState<uint32_t>(),
                         sizeof(uint32_t), cudaMemcpyHostToDevice));
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc,
                 const Workspace &ws) override { return false; }

  void RunImpl(Workspace &ws) override {}

  uint32_t GetState() const {
    uint32_t ret;
    state_.order.wait(AccessOrder::host());
    CUDA_CALL(cudaMemcpy(&ret, state_.ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return ret;
  }

 private:
  DummyGPUData state_;
};

DALI_REGISTER_OPERATOR(DummySource, DummyOperatorWithState<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(DummySource, DummyOperatorWithState<GPUBackend>, Mixed);
DALI_REGISTER_OPERATOR(DummySource, DummyOperatorWithState<GPUBackend>, GPU);

DALI_SCHEMA(DummySource)
  .DocStr("Dummy")
  .NumInput(0)
  .NumOutput(2)
  .AddArg("dummy_state", "internal dummy state", DALI_UINT32);

DALI_REGISTER_OPERATOR(DummyInnerLayer, DummyOperatorWithState<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(DummyInnerLayer, DummyOperatorWithState<GPUBackend>, Mixed);
DALI_REGISTER_OPERATOR(DummyInnerLayer, DummyOperatorWithState<GPUBackend>, GPU);

DALI_SCHEMA(DummyInnerLayer)
  .DocStr("Dummy")
  .NumInput(1)
  .NumOutput(1)
  .AddArg("dummy_state", "internal dummy state", DALI_UINT32);

DALI_REGISTER_OPERATOR(DummyOutput, DummyOperatorWithState<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(DummyOutput, DummyOperatorWithState<GPUBackend>, Mixed);
DALI_REGISTER_OPERATOR(DummyOutput, DummyOperatorWithState<GPUBackend>, GPU);

DALI_SCHEMA(DummyOutput)
  .DocStr("Dummy")
  .NumInput(2)
  .NumOutput(1)
  .AddArg("dummy_state", "internal dummy state", DALI_UINT32);

class CheckpointTest : public DALITest {
 public:
  CheckpointTest() : stream_(CUDAStreamPool::instance().Get()) {}

  inline OpSpec& PrepareSpec(OpSpec &spec) {
    spec.AddArg("max_batch_size", 1)
      .AddArg("num_threads", 1)
      .AddArg("cuda_stream", (cudaStream_t) this->stream_.get())
      .AddArg("pixels_per_image_hint", 0);
    return spec;
  }

  uint32_t GetDummyState(const OperatorBase &dummy_op) {
    try {
      return GetDummyStateTyped<CPUBackend>(dummy_op);
    } catch(const std::bad_cast &) {
      try {
        return GetDummyStateTyped<GPUBackend>(dummy_op);
      } catch(const std::bad_cast &) {
        ADD_FAILURE() << "Passed OperatorBase in not a dummy operator.";
        return 0;
      }
    }
  }

  enum OperatorStatesPolicy {
    ZERO_STATE,
    UNIQUE_STATES,
  };

  using GraphFactory = std::function<OpGraph(OperatorStatesPolicy)>;

  void RunTestOnGraph(GraphFactory make_graph_instance) {
    Checkpoint checkpoint;

    OpGraph original_graph = make_graph_instance(UNIQUE_STATES);
    BuildFromLegacyGraph(checkpoint, original_graph);

    int nodes_cnt = original_graph.NumOp();

    OpGraph new_graph = make_graph_instance(ZERO_STATE);

    for (OpNodeId i = 0; i < nodes_cnt; i++) {
      const auto &name = original_graph.Node(i).instance_name;
      ASSERT_EQ(name,
                checkpoint.GetOpCheckpoint(i).OperatorName());
      EXPECT_EQ(&checkpoint.GetOpCheckpoint(i), &checkpoint.GetOpCheckpoint(name));
      original_graph.Node(i).op->SaveState(checkpoint.GetOpCheckpoint(i),
                                           AccessOrder(this->stream_.get()));
    }

    ASSERT_EQ(new_graph.NumOp(), nodes_cnt);
    for (OpNodeId i = 0; i < nodes_cnt; i++) {
      ASSERT_EQ(new_graph.Node(i).instance_name,
                checkpoint.GetOpCheckpoint(i).OperatorName());
      checkpoint.GetOpCheckpoint(i).SetOrder(AccessOrder::host());
      new_graph.Node(i).op->RestoreState(checkpoint.GetOpCheckpoint(i));
    }

    for (OpNodeId i = 0; i < nodes_cnt; i++)
      EXPECT_EQ(this->GetDummyState(*original_graph.Node(i).op),
                this->GetDummyState(*new_graph.Node(i).op));
  }

  uint32_t NextState(OperatorStatesPolicy policy) {
    switch (policy) {
      case UNIQUE_STATES:
        return this->counter_++;
      case ZERO_STATE:
        return 0;
      default:
        ADD_FAILURE() << "Invalid enum value.";
        return 0;
    }
  }

 private:
  template<class Backend>
  uint32_t GetDummyStateTyped(const OperatorBase &dummy_op) {
    return dynamic_cast<const DummyOperatorWithState<Backend> &>(dummy_op).GetState();
  }

  uint32_t counter_ = 0;
  CUDAStreamLease stream_;
};

TEST_F(CheckpointTest, CPUOnly) {
  this->RunTestOnGraph([this](OperatorStatesPolicy policy) {
    OpGraph graph;

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummySource")
            .AddArg("device", "cpu")
            .AddArg("dummy_state", this->NextState(policy))
            .AddOutput("data_node_1", StorageDevice::CPU)
            .AddOutput("data_node_2", StorageDevice::CPU)), "source");

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummyInnerLayer")
            .AddArg("device", "cpu")
            .AddArg("dummy_state", this->NextState(policy))
            .AddInput("data_node_1", StorageDevice::CPU)
            .AddOutput("data_node_3", StorageDevice::CPU)), "inner1");

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummyInnerLayer")
            .AddArg("device", "cpu")
            .AddArg("dummy_state", this->NextState(policy))
            .AddInput("data_node_2", StorageDevice::CPU)
            .AddOutput("data_node_4", StorageDevice::CPU)), "inner2");

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummyOutput")
            .AddArg("device", "cpu")
            .AddArg("dummy_state", this->NextState(policy))
            .AddInput("data_node_3", StorageDevice::CPU)
            .AddInput("data_node_4", StorageDevice::CPU)
            .AddOutput("data_output", StorageDevice::CPU)), "output");

    graph.InstantiateOperators();
    return graph;
  });
}

TEST_F(CheckpointTest, GPUOnly) {
  this->RunTestOnGraph([this](OperatorStatesPolicy policy) {
    OpGraph graph;

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummySource")
            .AddArg("device", "gpu")
            .AddArg("dummy_state", this->NextState(policy))
            .AddOutput("data_node_1", StorageDevice::GPU)
            .AddOutput("data_node_2", StorageDevice::GPU)), "source");

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummyInnerLayer")
            .AddArg("device", "gpu")
            .AddArg("dummy_state", this->NextState(policy))
            .AddInput("data_node_1", StorageDevice::GPU)
            .AddOutput("data_node_3", StorageDevice::GPU)), "inner1");

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummyInnerLayer")
            .AddArg("device", "gpu")
            .AddArg("dummy_state", this->NextState(policy))
            .AddInput("data_node_2", StorageDevice::GPU)
            .AddOutput("data_node_4", StorageDevice::GPU)), "inner2");

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummyOutput")
            .AddArg("device", "gpu")
            .AddArg("dummy_state", this->NextState(policy))
            .AddInput("data_node_3", StorageDevice::GPU)
            .AddInput("data_node_4", StorageDevice::GPU)
            .AddOutput("data_output", StorageDevice::GPU)), "output");

    graph.InstantiateOperators();
    return graph;
  });
}

TEST_F(CheckpointTest, Mixed) {
  this->RunTestOnGraph([this](OperatorStatesPolicy policy) {
    OpGraph graph;

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummySource")
            .AddArg("device", "cpu")
            .AddArg("dummy_state", this->NextState(policy))
            .AddOutput("data_node_1", StorageDevice::CPU)
            .AddOutput("data_node_2", StorageDevice::CPU)), "stateful_source");

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummyInnerLayer")
            .AddArg("device", "mixed")
            .AddArg("dummy_state", this->NextState(policy))
            .AddInput("data_node_1", StorageDevice::CPU)
            .AddOutput("data_node_3", StorageDevice::GPU)), "stateful_op_1");

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummyInnerLayer")
            .AddArg("device", "mixed")
            .AddArg("dummy_state", this->NextState(policy))
            .AddInput("data_node_2", StorageDevice::CPU)
            .AddOutput("data_node_4", StorageDevice::GPU)), "stateful_op_2");

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummyOutput")
            .AddArg("device", "gpu")
            .AddArg("dummy_state", this->NextState(policy))
            .AddInput("data_node_3", StorageDevice::GPU)
            .AddInput("data_node_4", StorageDevice::GPU)
            .AddOutput("data_output", StorageDevice::GPU)), "dummy_output");

    graph.InstantiateOperators();
    return graph;
  });
}

TEST_F(CheckpointTest, Serialize) {
  Checkpoint checkpoint;
  OpGraph graph;
  auto exec = GetSimpleExecutor();

  graph.AddOp(this->PrepareSpec(
    OpSpec("TestStatefulSource")
    .AddArg("device", "cpu")
    .AddArg("epoch_size", 1)
    .AddOutput("data_1", StorageDevice::CPU)), "stateful_source");

  graph.AddOp(this->PrepareSpec(
    OpSpec("TestStatefulOp")
    .AddArg("device", "cpu")
    .AddInput("data_1", StorageDevice::CPU)
    .AddOutput("data_2", StorageDevice::CPU)), "stateful_op_1");

  graph.AddOp(this->PrepareSpec(
    OpSpec("TestStatefulOp")
    .AddArg("device", "mixed")
    .AddInput("data_2", StorageDevice::CPU)
    .AddOutput("data_3", StorageDevice::GPU)), "stateful_op_2");

  graph.AddOp(this->PrepareSpec(
    OpSpec("TestStatefulOp")
    .AddArg("device", "gpu")
    .AddInput("data_3", StorageDevice::GPU)
    .AddOutput("data_4", StorageDevice::GPU)), "stateful_op_3");

  exec->Build(&graph, {"data_4_gpu"});
  BuildFromLegacyGraph(checkpoint, graph);

  size_t nodes = static_cast<size_t>(graph.NumOp());
  for (uint8_t i = 0; i < nodes; i++)
    checkpoint.GetOpCheckpoint(i).MutableCheckpointState() = i;

  auto serialized = checkpoint.SerializeToProtobuf(*exec);

  Checkpoint deserialized;
  deserialized.DeserializeFromProtobuf(*exec, serialized);

  ASSERT_EQ(deserialized.NumOp(), nodes);
  for (uint8_t i = 0; i < nodes; i++)
    EXPECT_EQ(deserialized.GetOpCheckpoint(i).CheckpointState<uint8_t>(), i);
}

}  // namespace dali
