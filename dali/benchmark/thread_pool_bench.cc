// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/benchmark/dali_bench.h"
#include "dali/pipeline/util/thread_pool.h"

namespace dali {

class ThreadPoolBench : public DALIBenchmark {};

static void ThreadPoolArgs(benchmark::internal::Benchmark *b) {
  int batch_size = 64;
  int work_size_min = 400;
  int work_size_max = 10000;
  int nthreads = 4;
  b->Args({batch_size, work_size_min, work_size_max, nthreads});
}

BENCHMARK_DEFINE_F(ThreadPoolBench, AddWork)(benchmark::State& st) {
  int batch_size = st.range(0);
  int work_size_min = st.range(1);
  int work_size_max = st.range(2);
  int nthreads = st.range(3);

  ThreadPool thread_pool(nthreads, 0, false);

  std::vector<uint8_t> data(2000, 0xFF);
  std::atomic<int64_t> total_count(0);
  while (st.KeepRunning()) {
    for (int i = 0; i < batch_size; i++) {
        auto size = this->RandInt(work_size_min, work_size_max);
        thread_pool.AddWork(
          [&data, size, &total_count](int thread_id){
            std::vector<uint8_t> other_data;
            for (int i = 0; i < size; i++) {
              other_data.push_back(data[i%data.size()]);
            }
            std::this_thread::sleep_for(std::chrono::nanoseconds(size * 10));
            total_count += size;
          }, 0, true);
    }
    thread_pool.WaitForWork();

    int num_batches = st.iterations() + 1;
    st.counters["FPS"] = benchmark::Counter(batch_size*num_batches,
        benchmark::Counter::kIsRate);
  }
  std::cout << total_count << std::endl;
}

BENCHMARK_REGISTER_F(ThreadPoolBench, AddWork)->Iterations(1000)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->Apply(ThreadPoolArgs);


BENCHMARK_DEFINE_F(ThreadPoolBench, AddWorkDeferred)(benchmark::State& st) {
  int batch_size = st.range(0);
  int work_size_min = st.range(1);
  int work_size_max = st.range(2);
  int nthreads = st.range(3);

  ThreadPool thread_pool(nthreads, 0, false);
  std::vector<uint8_t> data(2000, 0xFF);

  std::atomic<int64_t> total_count(0);
  while (st.KeepRunning()) {
    for (int i = 0; i < batch_size; i++) {
        auto size = this->RandInt(work_size_min, work_size_max);
        thread_pool.AddWork(
          [&data, size, &total_count](int thread_id){
            std::vector<uint8_t> other_data;
            for (int i = 0; i < size; i++) {
              other_data.push_back(data[i%data.size()]);
            }
            std::this_thread::sleep_for(std::chrono::nanoseconds(size * 10));
            total_count += size;
          }, size);
    }
    thread_pool.RunAll();

    int num_batches = st.iterations() + 1;
    st.counters["FPS"] = benchmark::Counter(batch_size*num_batches,
        benchmark::Counter::kIsRate);
  }
  std::cout << total_count << std::endl;
}


BENCHMARK_REGISTER_F(ThreadPoolBench, AddWorkDeferred)->Iterations(1000)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->Apply(ThreadPoolArgs);

}  // namespace dali
