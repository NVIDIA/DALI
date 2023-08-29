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
      , state_(spec.GetArgument<uint8_t>("dummy_state")) {}

  void SaveState(OpCheckpoint &cpt, std::optional<cudaStream_t> stream) override {
    cpt.MutableCheckpointState() = DummySnapshot{{state_}};
  }

  void RestoreState(const OpCheckpoint &cpt) override {
    state_ = cpt.CheckpointState<DummySnapshot>().dummy_state[0];
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc,
                 const Workspace &ws) override { return false; }

  uint8_t GetState() const { return state_; }

 private:
  uint8_t state_;
};

struct DummyGPUData {
  uint8_t *ptr;
  AccessOrder order;

  inline DummyGPUData(cudaStream_t stream, uint8_t value) : order(stream) {
    CUDA_CALL(cudaMalloc(&ptr, sizeof(uint8_t)));
    CUDA_CALL(cudaMemcpyAsync(ptr, &value, sizeof(uint8_t),
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
               spec.GetArgument<uint8_t>("dummy_state")) {}

  void SaveState(OpCheckpoint &cpt, std::optional<cudaStream_t> stream) override {
    if (!stream)
      FAIL() << "Cuda stream was not provided for GPU operator checkpointing.";

    CheckpointingData &cpt_state = cpt.MutableCheckpointState();
    if (!std::holds_alternative<DummySnapshot>(cpt_state))
      cpt_state = DummySnapshot{{0}};

    cpt.SetOrder(AccessOrder(*stream));
    CUDA_CALL(cudaMemcpyAsync(std::get<DummySnapshot>(cpt_state).dummy_state.data(),
                              state_.ptr, sizeof(uint8_t), cudaMemcpyDeviceToHost, *stream));
  }

  void RestoreState(const OpCheckpoint &cpt) override {
    CUDA_CALL(cudaMemcpy(state_.ptr, cpt.CheckpointState<DummySnapshot>().dummy_state.data(),
                         sizeof(uint8_t), cudaMemcpyHostToDevice));
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc,
                 const Workspace &ws) override { return false; }

  void RunImpl(Workspace &ws) override {}

  uint8_t GetState() const {
    uint8_t ret;
    state_.order.wait(AccessOrder::host());
    CUDA_CALL(cudaMemcpy(&ret, state_.ptr, sizeof(uint8_t), cudaMemcpyDeviceToHost));
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

  uint8_t GetDummyState(const OperatorBase &dummy_op) {
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

  uint8_t NextState(OperatorStatesPolicy policy) {
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
  uint8_t GetDummyStateTyped(const OperatorBase &dummy_op) {
    return dynamic_cast<const DummyOperatorWithState<Backend> &>(dummy_op).GetState();
  }

  uint8_t counter_ = 0;
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
  OpGraph graph;
  graph.AddOp(this->PrepareSpec(
          OpSpec("DummySource")
          .AddArg("device", "cpu")
          .AddArg("dummy_state", 0)
          .AddOutput("data1", "cpu")
          .AddOutput("data2", "cpu")), "");
  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyInnerLayer")
          .AddArg("device", "cpu")
          .AddArg("dummy_state", 1)
          .AddInput("data1", "cpu")
          .AddOutput("data3", "cpu")), "");
  graph.InstantiateOperators();

  std::string serialized_data = [&] {
    Checkpoint cpt;
    cpt.Build(graph);

    ASSERT_EQ(cpt.NumOp(), 2);
    ASSERT_EQ(cpt.GetOpCheckpoint(0).OperatorName(), "DummySource");
    ASSERT_EQ(cpt.GetOpCheckpoint(1).OperatorName(), "DummyInnerLayer");

    cpt.GetOpCheckpoint(0).MutableCheckpointState() = DummySnapshot{{0}};
    cpt.GetOpCheckpoint(1).MutableCheckpointState() = DummySnapshot{{1}};
    return cpt.SerializeToProtobuf();
  }();

  Checkpoint cpt;
  cpt.DeserializeFromProtobuf(serialized_data);

  ASSERT_EQ(cpt.NumOp(), 2);
  ASSERT_EQ(cpt.GetOpCheckpoint(0).OperatorName(), "DummySource");
  ASSERT_EQ(cpt.GetOpCheckpoint(1).OperatorName(), "DummyInnerLayer");
  ASSERT_TRUE(std::holds_alternative<DummySnapshot>(cpt.GetOpCheckpoint(0).GetCheckpointingData()));
  ASSERT_TRUE(std::holds_alternative<DummySnapshot>(cpt.GetOpCheckpoint(1).GetCheckpointingData()));
  EXPECT_EQ(
    cpt.GetOpCheckpoint(0).CheckpointState<DummySnapshot>().dummy_state,
    std::vector<uint8_t>{0});
  EXPECT_EQ(
    cpt.GetOpCheckpoint(1).CheckpointState<DummySnapshot>().dummy_state,
    std::vector<uint8_t>{1});
}

}  // namespace dali
