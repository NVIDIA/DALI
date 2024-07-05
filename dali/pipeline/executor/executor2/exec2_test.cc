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

#include "dali/pipeline/executor/executor2/exec2_test.h"
#include "dali/pipeline/executor/executor2/exec2.h"

namespace dali {

DALI_SCHEMA(Exec2TestOp)  // DALI_SCHEMA can't take a macro :(
  .NumInput(0, 99)
  .NumOutput(1)
  .AddArg("addend", "a value added to the sum of inputs", DALI_INT32, true);

// DALI_REGISTER_OPERATOR can't take a macro for the name
DALI_REGISTER_OPERATOR(Exec2TestOp, exec2::test::DummyOpCPU, CPU);

DALI_SCHEMA(Exec2Counter)
  .NumInput(0)
  .NumOutput(1);

DALI_REGISTER_OPERATOR(Exec2Counter, exec2::test::CounterOp, CPU);

namespace exec2 {
namespace test {

class Exec2Test : public::testing::TestWithParam<Executor2::Config> {
 public:
  Exec2Test() {
    config_ = GetParam();
  }

  Executor2::Config config_;
};


TEST_P(Exec2Test, SimpleGraph) {
  Executor2 exec(config_);
  graph::OpGraph graph = GetTestGraph1();
  exec.Build(graph);
  exec.Run();
  exec.Run();
  Workspace ws;
  exec.Outputs(&ws);
  CheckTestGraph1Results(ws, config_.max_batch_size);
  ws.Clear();
  exec.Outputs(&ws);
  CheckTestGraph1Results(ws, config_.max_batch_size);
}

Executor2::Config MakeCfg(QueueDepthPolicy q, OperatorConcurrency c, StreamPolicy s) {
  Executor2::Config cfg;
  cfg.queue_policy = q;
  cfg.concurrency = c;
  cfg.stream_policy = s;
  cfg.thread_pool_threads = 4;
  cfg.operator_threads = 4;
  return cfg;
}

std::vector<Executor2::Config> configs = {
  MakeCfg(QueueDepthPolicy::OutputOnly, OperatorConcurrency::None, StreamPolicy::Single),
  MakeCfg(QueueDepthPolicy::BackendChange, OperatorConcurrency::Backend, StreamPolicy::PerBackend),
  MakeCfg(QueueDepthPolicy::FullyBuffered, OperatorConcurrency::Full, StreamPolicy::PerOperator),
};

INSTANTIATE_TEST_SUITE_P(Exec2Test, Exec2Test, testing::ValuesIn(configs));


}  // namespace test
}  // namespace exec2
}  // namespace dali
