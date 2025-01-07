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

#include <benchmark/benchmark.h>

#include "dali/benchmark/dali_bench.h"
#include "dali/pipeline/pipeline.h"
#include "dali/util/image.h"
#include "dali/test/dali_test_config.h"

namespace dali {

enum class CheckpointingPolicy {Disabled, Enabled, SaveEveryIter, SerializeEveryIter};

class CheckpointingOverhead : public DALIBenchmark {
 public:
  void run(benchmark::State& st,
           const OpSpec &op_spec,
           const vector<std::pair<string, string>> &outputs) {
    auto policy = static_cast<CheckpointingPolicy>(st.range(0));
    bool checkpointing = (policy != CheckpointingPolicy::Disabled);

    auto pipe = createPipeline(op_spec);
    if (checkpointing) {
      pipe->EnableCheckpointing();
    }
    pipe->Build(outputs);

    Workspace ws;

    // Warmup
    pipe->Run();
    pipe->Outputs(&ws);

    while (st.KeepRunning()) {
      pipe->Run();
      pipe->Outputs(&ws);
      if (policy == CheckpointingPolicy::SaveEveryIter) {
        volatile auto cpt = pipe->GetCheckpoint();
      } else if (policy == CheckpointingPolicy::SerializeEveryIter) {
        volatile auto cpt = pipe->SerializedCheckpoint({});
      }
    }

    if (policy == CheckpointingPolicy::Disabled) {
      st.SetLabel("disabled");
    } else if (policy == CheckpointingPolicy::Enabled) {
      st.SetLabel("enabled");
    } else if (policy == CheckpointingPolicy::SaveEveryIter) {
      st.SetLabel("save");
    } else if (policy == CheckpointingPolicy::SerializeEveryIter) {
      st.SetLabel("serialize");
    }
  }

 protected:
  std::unique_ptr<Pipeline> createPipeline(const OpSpec &op_spec) {
    const int batch_size = 256;
    const int num_thread = 4;
    const bool pipelined = true;
    const int prefetch_queue_depth = 2;
    const bool async = true;

    auto pipe = std::make_unique<Pipeline>(batch_size, num_thread, 0, -1, pipelined,
                                           prefetch_queue_depth, async);
    pipe->AddOperator(op_spec);
    return pipe;
  }
};

static void Args(benchmark::internal::Benchmark *b) {
  const std::vector<CheckpointingPolicy> policies = {
    CheckpointingPolicy::Disabled,
    CheckpointingPolicy::Enabled,
    CheckpointingPolicy::SaveEveryIter,
    CheckpointingPolicy::SerializeEveryIter,
  };
  for (auto p : policies) {
    b->Args({static_cast<int>(p)});
  }
}

BENCHMARK_DEFINE_F(CheckpointingOverhead, StatelessCpu)(benchmark::State& st) {
  auto op = OpSpec("Constant")
        .AddArg("device", "cpu")
        .AddArg("idata", std::vector<int>(1, 1))
        .AddArg("shape", std::vector<int>(2, 100))  // 100x100
        .AddOutput("output", StorageDevice::CPU);
  this->run(st, op, {{"output", "cpu"}});
}

BENCHMARK_REGISTER_F(CheckpointingOverhead, StatelessCpu)->Iterations(100)
->Unit(benchmark::kMillisecond)
->Apply(Args);

BENCHMARK_DEFINE_F(CheckpointingOverhead, StatelessGpu)(benchmark::State& st) {
  auto op = OpSpec("Constant")
        .AddArg("device", "gpu")
        .AddArg("idata", std::vector<int>(1, 1))
        .AddArg("shape", std::vector<int>(2, 100))  // 100x100
        .AddOutput("output", StorageDevice::GPU);
  this->run(st, op, {{"output", "gpu"}});
}

BENCHMARK_REGISTER_F(CheckpointingOverhead, StatelessGpu)->Iterations(100)
->Unit(benchmark::kMillisecond)
->Apply(Args);

BENCHMARK_DEFINE_F(CheckpointingOverhead, RandomCpu)(benchmark::State& st) {
  auto op = OpSpec("CoinFlip")
        .AddArg("device", "cpu")
        .AddArg("shape", std::vector<int>(2, 100))  // 100x100
        .AddOutput("output", StorageDevice::CPU);
  this->run(st, op, {{"output", "cpu"}});
}

BENCHMARK_REGISTER_F(CheckpointingOverhead, RandomCpu)->Iterations(100)
->Unit(benchmark::kMillisecond)
->Apply(Args);

BENCHMARK_DEFINE_F(CheckpointingOverhead, RandomGpu)(benchmark::State& st) {
  auto op = OpSpec("CoinFlip")
        .AddArg("device", "gpu")
        .AddArg("shape", std::vector<int>(2, 100))  // 100x100
        .AddOutput("output", StorageDevice::GPU);
  this->run(st, op, {{"output", "gpu"}});
}

BENCHMARK_REGISTER_F(CheckpointingOverhead, RandomGpu)->Iterations(100)
->Unit(benchmark::kMillisecond)
->Apply(Args);

BENCHMARK_DEFINE_F(CheckpointingOverhead, Reader)(benchmark::State& st) {
  auto op = OpSpec("FileReader")
        .AddArg("device", "cpu")
        .AddArg("files", jpeg_names_)
        .AddArg("initial_fill", 1024)
        .AddArg("random_shuffle", true)
        .AddOutput("output", StorageDevice::CPU)
        .AddOutput("labels", StorageDevice::CPU);
  this->run(st, op, {{"output", "cpu"}});
}

BENCHMARK_REGISTER_F(CheckpointingOverhead, Reader)->Iterations(100)
->Unit(benchmark::kMillisecond)
->Apply(Args);

}  // namespace dali
