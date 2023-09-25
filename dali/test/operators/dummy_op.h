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

#ifndef DALI_TEST_OPERATORS_DUMMY_OP_H_
#define DALI_TEST_OPERATORS_DUMMY_OP_H_

#include <string>
#include <vector>

#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class DummyOp : public Operator<Backend> {
 public:
  inline explicit DummyOp(const OpSpec &spec) :
    Operator<Backend>(spec) {}

  inline ~DummyOp() override = default;

  DISABLE_COPY_MOVE_ASSIGN(DummyOp);

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return false;
  }

  void RunImpl(Workspace &) override {
    DALI_FAIL("I'm a dummy op don't run me");
  }
};

class TestStatefulSource : public Operator<CPUBackend> {
 public:
  explicit TestStatefulSource(const OpSpec &spec);

  inline ~TestStatefulSource() override = default;

  DISABLE_COPY_MOVE_ASSIGN(TestStatefulSource);

  void SaveState(OpCheckpoint &cpt, std::optional<cudaStream_t> stream) override;

  void RestoreState(const OpCheckpoint &cpt) override;

  std::string SerializeCheckpoint(const OpCheckpoint &cpt) const override;

  void DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) const override;

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;

  using Operator<CPUBackend>::RunImpl;

  void RunImpl(Workspace &ws) override;

 private:
  uint8_t state_ = 0;
  int checkpoints_to_collect_ = 1;
  int epoch_size_;
};

class TestStatefulOpCPU : public Operator<CPUBackend> {
 public:
  explicit TestStatefulOpCPU(const OpSpec &spec);

  inline ~TestStatefulOpCPU() override = default;

  DISABLE_COPY_MOVE_ASSIGN(TestStatefulOpCPU);

  void SaveState(OpCheckpoint &cpt, std::optional<cudaStream_t> stream) override;

  void RestoreState(const OpCheckpoint &cpt) override;

  std::string SerializeCheckpoint(const OpCheckpoint &cpt) const override;

  void DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) const override;

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;

  using Operator<CPUBackend>::RunImpl;

  void RunImpl(Workspace &ws) override;

 private:
  uint8_t state_ = 0;
};

class TestStatefulOpMixed : public Operator<MixedBackend> {
 public:
  explicit TestStatefulOpMixed(const OpSpec &spec);

  inline ~TestStatefulOpMixed() override = default;

  DISABLE_COPY_MOVE_ASSIGN(TestStatefulOpMixed);

  void SaveState(OpCheckpoint &cpt, std::optional<cudaStream_t> stream) override;

  void RestoreState(const OpCheckpoint &cpt) override;

  std::string SerializeCheckpoint(const OpCheckpoint &cpt) const override;

  void DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) const override;

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;

  void Run(Workspace &ws) override;

 private:
  uint8_t state_ = 0;
};

class TestStatefulOpGPU : public Operator<GPUBackend> {
 public:
  explicit TestStatefulOpGPU(const OpSpec &spec);

  inline ~TestStatefulOpGPU() override;

  DISABLE_COPY_MOVE_ASSIGN(TestStatefulOpGPU);

  void SaveState(OpCheckpoint &cpt, std::optional<cudaStream_t> stream) override;

  void RestoreState(const OpCheckpoint &cpt) override;

  std::string SerializeCheckpoint(const OpCheckpoint &cpt) const override;

  void DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) const override;

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;

  void RunImpl(Workspace &ws) override;

 private:
  int max_batch_size_;
  uint8_t *state_;
};

}  // namespace dali

#endif  // DALI_TEST_OPERATORS_DUMMY_OP_H_
