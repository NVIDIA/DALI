// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/test/operators/dummy_op.h"

#include <cstdlib>
#include <string>

namespace dali {

TestStatefulSource::TestStatefulSource(const OpSpec &spec)
    : Operator<CPUBackend>(spec),
      epoch_size_(spec.GetArgument<int>("epoch_size")) {}

void TestStatefulSource::SaveState(OpCheckpoint &cpt, std::optional<cudaStream_t> stream) {
  DALI_ENFORCE(checkpoints_to_collect_ > 0,
               "Attempting to collect a checkpoint from an empty queue. ");
  checkpoints_to_collect_--;  /* simulate removing checkpoint from queue */
  cpt.MutableCheckpointState() = state_;
}

void TestStatefulSource::RestoreState(const OpCheckpoint &cpt) {
  state_ = cpt.CheckpointState<uint8_t>();
}

std::string TestStatefulSource::SerializeCheckpoint(const OpCheckpoint &cpt) const {
  return std::to_string(cpt.CheckpointState<uint8_t>());
}

void TestStatefulSource::DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) const {
  cpt.MutableCheckpointState() = static_cast<uint8_t>(std::stoi(data));
}

bool TestStatefulSource::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  return false;
}

void TestStatefulSource::RunImpl(Workspace &ws) {
  auto &output = ws.Output<CPUBackend>(0);
  int samples = ws.GetRequestedBatchSize(0);
  output.set_type(DALI_UINT8);
  output.Resize(uniform_list_shape(samples, {1}));

  /* Return increasing integers as the samples, padding the last batch */
  for (int i = 0; i < samples; i++) {
    state_ = (state_ < epoch_size_ ? state_ : epoch_size_ - 1);
    output.mutable_tensor<uint8_t>(i)[0] = state_++;
  }

  /* Check if a new epoch starts in the next iteration, just like in Readers. */
  if (state_ == epoch_size_) {
    state_ = 0;
    checkpoints_to_collect_++;  /* simulate putting a checkpoint into a queue */
  }
}

TestStatefulOpCPU::TestStatefulOpCPU(const OpSpec &spec)
    : Operator<CPUBackend>(spec) {}

void TestStatefulOpCPU::SaveState(OpCheckpoint &cpt, std::optional<cudaStream_t> stream) {
  cpt.MutableCheckpointState() = state_;
}

void TestStatefulOpCPU::RestoreState(const OpCheckpoint &cpt) {
  state_ = cpt.CheckpointState<uint8_t>();
}

std::string TestStatefulOpCPU::SerializeCheckpoint(const OpCheckpoint &cpt) const {
  return std::to_string(cpt.CheckpointState<uint8_t>());
}

void TestStatefulOpCPU::DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) const {
  cpt.MutableCheckpointState() = static_cast<uint8_t>(std::stoi(data));
}

bool TestStatefulOpCPU::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  return false;
}

void TestStatefulOpCPU::RunImpl(Workspace &ws) {
  auto &input = ws.Input<CPUBackend>(0);
  auto &output = ws.Output<CPUBackend>(0);
  int samples = input.num_samples();
  output.set_type(DALI_UINT8);
  output.Resize(uniform_list_shape(samples, {1}));

  for (int i = 0; i < samples; i++)
    output.mutable_tensor<uint8_t>(i)[0] = input.tensor<uint8_t>(i)[0] + state_++;
}

DALI_REGISTER_OPERATOR(DummyOp, DummyOp<CPUBackend>, CPU);

DALI_SCHEMA(DummyOp)
    .DocStr("Dummy operator for testing")
    .OutputFn([](const OpSpec &spec) { return spec.GetArgument<int>("num_outputs"); })
    .NumInput(0, 10)
    .AddOptionalArg("num_outputs", R"code(Number of outputs.)code", 2)
    .AddOptionalArg("arg_input_f", "Float argument input used for tests", 0.f, true)
    .AddOptionalArg("arg_input_i", "Integer argument input used for tests", 0, true);


DALI_REGISTER_OPERATOR(TestStatefulSource, TestStatefulSource, CPU);
DALI_SCHEMA(TestStatefulSource)
  .DocStr("Simple source operator for checkpointing testing")
  .AddArg("epoch_size", "Testing epoch size", DALI_INT32)
  .NumInput(0)
  .NumOutput(1);

DALI_REGISTER_OPERATOR(TestStatefulOp, TestStatefulOpCPU, CPU);
DALI_SCHEMA(TestStatefulOp)
  .DocStr("Simple operator for checkpointing testing")
  .NumInput(1)
  .NumOutput(1);

}  // namespace dali
