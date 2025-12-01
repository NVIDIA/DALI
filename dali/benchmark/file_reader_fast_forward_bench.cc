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

class FileReaderFastForward : public DALIBenchmark {
 public:
  // Returns a vector of n (possibly repeating) paths to test images
  std::vector<std::string> get_paths(size_t n) {
    static std::vector<std::string> pool;
    auto jpegs = jpeg_names_;
    DALI_ENFORCE(!jpeg_names_.empty(), "No images!");
    while (pool.size() < n) {
      pool.insert(pool.end(), jpegs.begin(), jpegs.end());
    }
    return {pool.begin(), pool.begin() + n};
  }

  std::unique_ptr<Pipeline> create_reader_pipeline(size_t dataset_size, int initial_buffer_fill) {
    const int batch_size = 1;
    const int num_thread = 4;
    const bool pipelined = true;
    const int prefetch_queue_depth = 2;
    const bool async = true;
    bool random_shuffle = (initial_buffer_fill > 1);
    auto pipe = std::make_unique<Pipeline>(batch_size, num_thread, 0, -1, pipelined,
                                           prefetch_queue_depth, async);

    pipe->AddOperator(
        OpSpec("FileReader")
        .AddArg("device", "cpu")
        .AddArg("files", get_paths(dataset_size))
        .AddArg("initial_fill", initial_buffer_fill)
        .AddArg("random_shuffle", random_shuffle)
        .AddOutput("jpegs", StorageDevice::CPU)
        .AddOutput("labels", StorageDevice::CPU));

    pipe->EnableCheckpointing();

    return pipe;
  }
};

BENCHMARK_DEFINE_F(FileReaderFastForward, FastForward)(benchmark::State& st) { // NOLINT
  int dataset_size = st.range(0) + 1;
  int snapshot_at = st.range(1);
  int initial_buffer_fill = st.range(2);
  auto pipe = create_reader_pipeline(dataset_size, initial_buffer_fill);

  vector<std::pair<string, string>> outputs = {{"jpegs", "cpu"}};
  pipe->Build(outputs);

  Workspace ws;
  for (int i = 0; i < snapshot_at; i++) {
    pipe->Run();
    pipe->Outputs(&ws);
  }

  auto cpt = pipe->GetCheckpoint();
  while (st.KeepRunning()) {
    st.PauseTiming();
    auto pipe2 = create_reader_pipeline(dataset_size, initial_buffer_fill);
    pipe2->Build(outputs);
    st.ResumeTiming();

    pipe2->RestoreFromCheckpoint(cpt);

    st.PauseTiming();
    pipe2->Run();
    pipe2->Outputs(&ws);
    st.ResumeTiming();
  }
}

static void Args(benchmark::internal::Benchmark *b) {
  for (int dataset_size = 100; dataset_size <= 1000000; dataset_size *= 100) {
    for (int frac = 0; frac <= 10; frac++) {
      // Without random shuffle
      b->Args({dataset_size, dataset_size * frac / 10, 1});

      // With random shuffle, initial_fill=1024
      b->Args({dataset_size, dataset_size * frac / 10, 1024});
    }
  }
}

BENCHMARK_REGISTER_F(FileReaderFastForward, FastForward)->Iterations(1)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(Args);

}  // namespace dali
