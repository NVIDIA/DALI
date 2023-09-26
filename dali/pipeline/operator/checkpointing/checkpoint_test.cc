// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/pipeline/graph/op_graph.h"

namespace dali {

template <typename Backend>
class DummyOperatorWithState : public Operator<Backend> {};

template<>
class DummyOperatorWithState<CPUBackend> : public Operator<CPUBackend> {
 public:
  explicit DummyOperatorWithState(const OpSpec &spec)
      : Operator<CPUBackend>(spec)
      , state_(spec.GetArgument<uint32_t>("dummy_state")) {}

  void SaveState(OpCheckpoint &cpt, std::optional<cudaStream_t> stream) override {
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

  void SaveState(OpCheckpoint &cpt, std::optional<cudaStream_t> stream) override {
    if (!stream)
      FAIL() << "Cuda stream was not provided for GPU operator checkpointing.";

    std::any &cpt_state = cpt.MutableCheckpointState();
    if (!cpt_state.has_value())
      cpt_state = static_cast<uint32_t>(0);

    cpt.SetOrder(AccessOrder(*stream));
    CUDA_CALL(cudaMemcpyAsync(&std::any_cast<uint32_t &>(cpt_state), state_.ptr,
                              sizeof(uint32_t), cudaMemcpyDeviceToHost, *stream));
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
    checkpoint.Build(original_graph);

    int nodes_cnt = original_graph.NumOp();

    OpGraph new_graph = make_graph_instance(ZERO_STATE);

    for (OpNodeId i = 0; i < nodes_cnt; i++) {
      ASSERT_EQ(original_graph.Node(i).spec.name(),
                checkpoint.GetOpCheckpoint(i).OperatorName());
      original_graph.Node(i).op->SaveState(checkpoint.GetOpCheckpoint(i), this->stream_.get());
    }

    ASSERT_EQ(new_graph.NumOp(), nodes_cnt);
    for (OpNodeId i = 0; i < nodes_cnt; i++) {
      ASSERT_EQ(new_graph.Node(i).spec.name(),
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
            .AddOutput("data_node_1", "cpu")
            .AddOutput("data_node_2", "cpu")), "");

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummyInnerLayer")
            .AddArg("device", "cpu")
            .AddArg("dummy_state", this->NextState(policy))
            .AddInput("data_node_1", "cpu")
            .AddOutput("data_node_3", "cpu")), "");

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummyInnerLayer")
            .AddArg("device", "cpu")
            .AddArg("dummy_state", this->NextState(policy))
            .AddInput("data_node_2", "cpu")
            .AddOutput("data_node_4", "cpu")), "");

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummyOutput")
            .AddArg("device", "cpu")
            .AddArg("dummy_state", this->NextState(policy))
            .AddInput("data_node_3", "cpu")
            .AddInput("data_node_4", "cpu")
            .AddOutput("data_output", "cpu")), "");

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
            .AddOutput("data_node_1", "gpu")
            .AddOutput("data_node_2", "gpu")), "");

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummyInnerLayer")
            .AddArg("device", "gpu")
            .AddArg("dummy_state", this->NextState(policy))
            .AddInput("data_node_1", "gpu")
            .AddOutput("data_node_3", "gpu")), "");

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummyInnerLayer")
            .AddArg("device", "gpu")
            .AddArg("dummy_state", this->NextState(policy))
            .AddInput("data_node_2", "gpu")
            .AddOutput("data_node_4", "gpu")), "");

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummyOutput")
            .AddArg("device", "gpu")
            .AddArg("dummy_state", this->NextState(policy))
            .AddInput("data_node_3", "gpu")
            .AddInput("data_node_4", "gpu")
            .AddOutput("data_output", "gpu")), "");

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
            .AddOutput("data_node_1", "cpu")
            .AddOutput("data_node_2", "cpu")), "");

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummyInnerLayer")
            .AddArg("device", "mixed")
            .AddArg("dummy_state", this->NextState(policy))
            .AddInput("data_node_1", "cpu")
            .AddOutput("data_node_3", "gpu")), "");

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummyInnerLayer")
            .AddArg("device", "mixed")
            .AddArg("dummy_state", this->NextState(policy))
            .AddInput("data_node_2", "cpu")
            .AddOutput("data_node_4", "gpu")), "");

    graph.AddOp(this->PrepareSpec(
            OpSpec("DummyOutput")
            .AddArg("device", "gpu")
            .AddArg("dummy_state", this->NextState(policy))
            .AddInput("data_node_3", "gpu")
            .AddInput("data_node_4", "gpu")
            .AddOutput("data_output", "gpu")), "");

    graph.InstantiateOperators();
    return graph;
  });
}

TEST_F(CheckpointTest, Serialize) {
  Checkpoint checkpoint;
  OpGraph graph;

  graph.AddOp(this->PrepareSpec(
    OpSpec("TestStatefulSource")
    .AddArg("device", "cpu")
    .AddArg("epoch_size", 1)
    .AddOutput("data_1", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
    OpSpec("TestStatefulOp")
    .AddArg("device", "cpu")
    .AddInput("data_1", "cpu")
    .AddOutput("data_2", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
    OpSpec("TestStatefulOp")
    .AddArg("device", "mixed")
    .AddInput("data_2", "cpu")
    .AddOutput("data_3", "gpu")), "");

  graph.AddOp(this->PrepareSpec(
    OpSpec("TestStatefulOp")
    .AddArg("device", "gpu")
    .AddInput("data_3", "gpu")
    .AddOutput("data_4", "gpu")), "");

  graph.InstantiateOperators();
  checkpoint.Build(graph);

  size_t nodes = static_cast<size_t>(graph.NumOp());
  for (uint8_t i = 0; i < nodes; i++)
    checkpoint.GetOpCheckpoint(i).MutableCheckpointState() = i;

  auto serialized = checkpoint.SerializeToProtobuf(graph);

  Checkpoint deserialized;
  deserialized.DeserializeFromProtobuf(graph, serialized);

  ASSERT_EQ(deserialized.NumOp(), nodes);
  for (uint8_t i = 0; i < nodes; i++)
    EXPECT_EQ(deserialized.GetOpCheckpoint(i).CheckpointState<uint8_t>(), i);
}

}  // namespace dali
