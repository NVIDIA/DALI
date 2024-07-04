// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>
#include <vector>

namespace dali {

TestStatefulOpMixed::TestStatefulOpMixed(const OpSpec &spec)
    : Operator<MixedBackend>(spec) {}

void TestStatefulOpMixed::SaveState(OpCheckpoint &cpt, AccessOrder order) {
  cpt.MutableCheckpointState() = state_;
}

void TestStatefulOpMixed::RestoreState(const OpCheckpoint &cpt) {
  state_ = cpt.CheckpointState<uint8_t>();
}

std::string TestStatefulOpMixed::SerializeCheckpoint(const OpCheckpoint &cpt) const {
  return std::to_string(cpt.CheckpointState<uint8_t>());
}

void TestStatefulOpMixed::DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) const {
  cpt.MutableCheckpointState() = static_cast<uint8_t>(std::stoi(data));
}

bool TestStatefulOpMixed::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  return false;
}

void TestStatefulOpMixed::RunImpl(Workspace &ws) {
  auto &input = ws.Input<CPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  int samples = input.num_samples();
  output.set_type(DALI_UINT8);
  output.Resize(uniform_list_shape(samples, {1}));

  std::vector<uint8_t> buffer(samples);
  for (int i = 0; i < samples; i++) {
    buffer[i] = input.tensor<uint8_t>(i)[0] + state_++;
    CUDA_CALL(cudaMemcpyAsync(output.mutable_tensor<uint8_t>(i), &buffer[i], sizeof(uint8_t),
                              cudaMemcpyHostToDevice, ws.stream()));
  }
}

TestStatefulOpGPU::TestStatefulOpGPU(const OpSpec &spec)
    : Operator<GPUBackend>(spec) {
  max_batch_size_ = spec.GetArgument<int>("max_batch_size");
  CUDA_CALL(cudaMalloc(&state_, sizeof(uint8_t) * max_batch_size_));
  CUDA_CALL(cudaMemset(state_, 0, sizeof(uint8_t) * max_batch_size_));
}

TestStatefulOpGPU::~TestStatefulOpGPU() {
  CUDA_CALL(cudaFree(state_));
}

void TestStatefulOpGPU::SaveState(OpCheckpoint &cpt, AccessOrder order) {
  DALI_ENFORCE(order.is_device(), "Cuda stream was not provided for GPU operator checkpointing. ");

  std::any &cpt_state = cpt.MutableCheckpointState();
  if (!cpt_state.has_value())
    cpt_state = std::vector<uint8_t>(max_batch_size_);

  cpt.SetOrder(order);
  CUDA_CALL(cudaMemcpyAsync(std::any_cast<std::vector<uint8_t>>(cpt_state).data(),
                            state_, sizeof(uint8_t) * max_batch_size_,
                            cudaMemcpyDeviceToHost, order.stream()));
}

void TestStatefulOpGPU::RestoreState(const OpCheckpoint &cpt) {
  CUDA_CALL(cudaMemcpy(state_, cpt.CheckpointState<std::vector<uint8_t>>().data(),
                       sizeof(uint8_t) * max_batch_size_, cudaMemcpyHostToDevice));
}

std::string TestStatefulOpGPU::SerializeCheckpoint(const OpCheckpoint &cpt) const {
  return std::to_string(cpt.CheckpointState<uint8_t>());
}

void TestStatefulOpGPU::DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) const {
  cpt.MutableCheckpointState() = static_cast<uint8_t>(std::stoi(data));
}

bool TestStatefulOpGPU::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  return false;
}

__global__ void inc(uint8_t *state, uint8_t *output, const uint8_t *input) {
  *output = *input + *state++;
}

void TestStatefulOpGPU::RunImpl(Workspace &ws) {
  auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  int samples = input.num_samples();
  output.Resize(uniform_list_shape(samples, {1}), DALI_UINT8, BatchContiguity::Contiguous);

  for (int i = 0; i < samples; i++) {
    inc<<<1, 1, 0, ws.stream()>>>(
      state_ + i, output.mutable_tensor<uint8_t>(i), input.tensor<uint8_t>(i));
    CUDA_CALL(cudaGetLastError());
  }
}

DALI_REGISTER_OPERATOR(DummyOp, DummyOp<GPUBackend>, GPU);

DALI_REGISTER_OPERATOR(TestStatefulOp, TestStatefulOpMixed, Mixed);
DALI_REGISTER_OPERATOR(TestStatefulOp, TestStatefulOpGPU, GPU);

}  // namespace dali
