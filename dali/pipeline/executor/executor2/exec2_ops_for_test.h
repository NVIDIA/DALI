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

#ifndef DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC2_OPS_FOR_TEST_H_
#define DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC2_OPS_FOR_TEST_H_

#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/graph/op_graph2.h"

namespace dali {
namespace exec2 {
namespace test {

constexpr char kTestOpName[] = "Exec2TestOp";

class DummyOpCPU : public Operator<CPUBackend> {
 public:
  explicit DummyOpCPU(const OpSpec &spec) : Operator<CPUBackend>(spec) {
    instance_name_ = spec_.GetArgument<string>("name");
  }

  bool SetupImpl(std::vector<OutputDesc> &outs, const Workspace &ws) override {
    int N = ws.GetRequestedBatchSize(0);
    outs.resize(ws.NumOutput());
    outs[0].shape = uniform_list_shape(N, TensorShape<>{});
    outs[0].type = DALI_INT32;
    return true;
  }

  void RunImpl(Workspace &ws) override {
    int N = ws.GetRequestedBatchSize(0);
    addend_.Acquire(spec_, ws, N);
    sample_sums_.resize(N);
    auto &tp = ws.GetThreadPool();
    for (int s = 0; s < N; s++) {
      auto sample_sum = [&, s](int) {
        int sum = *addend_[s].data + s;
        for (int i = 0; i < ws.NumInput(); i++) {
          sum += *ws.Input<CPUBackend>(i)[s].data<int>();
        }
        sample_sums_[s] = sum;
      };
      tp.AddWork(sample_sum);
    }
    tp.RunAll(true);
    for (int s = 0; s < N; s++)
      *ws.Output<CPUBackend>(0)[s].mutable_data<int>() = sample_sums_[s];
  }

  bool CanInferOutputs() const override { return true; }
  ArgValue<int> addend_{"addend", spec_};

  std::vector<int> sample_sums_;
  std::string instance_name_;
};

class DummyOpGPU : public Operator<GPUBackend> {
 public:
  explicit DummyOpGPU(const OpSpec &spec) : Operator<GPUBackend>(spec) {
    instance_name_ = spec_.GetArgument<string>("name");
  }

  bool SetupImpl(std::vector<OutputDesc> &outs, const Workspace &ws) override {
    int N = ws.GetRequestedBatchSize(0);
    outs.resize(ws.NumOutput());
    outs[0].shape = uniform_list_shape(N, TensorShape<>{});
    outs[0].type = DALI_INT32;
    return true;
  }

  void RunImpl(Workspace &ws) override;

  bool CanInferOutputs() const override { return true; }

 private:
  ArgValue<int> addend_{"addend", spec_};

  std::string instance_name_;
};


constexpr char kCounterOpName[] = "Exec2Counter";

class CounterOp : public Operator<CPUBackend> {
 public:
  explicit CounterOp(const OpSpec &spec) : Operator<CPUBackend>(spec) {
  }

  bool SetupImpl(std::vector<OutputDesc> &outs, const Workspace &ws) override {
    int N = ws.GetRequestedBatchSize(0);
    outs.resize(ws.NumOutput());
    outs[0].shape = uniform_list_shape(N, TensorShape<>{});
    outs[0].type = DALI_INT32;
    return true;
  }

  void RunImpl(Workspace &ws) override {
    int N = ws.GetRequestedBatchSize(0);
    for (int s = 0; s < N; s++) {
      *ws.Output<CPUBackend>(0)[s].mutable_data<int>() = counter++;
    }
  }

  bool CanInferOutputs() const override { return true; }

  int counter = 0;
};

}  // namespace test
}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC2_OPS_FOR_TEST_H_
