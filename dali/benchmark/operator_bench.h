// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_BENCHMARK_OPERATOR_BENCH_H_
#define DALI_BENCHMARK_OPERATOR_BENCH_H_

#include <benchmark/benchmark.h>
#include "dali/benchmark/dali_bench.h"

namespace dali {

class OperatorBench : public DALIBenchmark {
 public:
  template <typename T>
  void RunCPU(benchmark::State& st, OpSpec op_spec,
              int W = 1920, int H = 1080, int C = 3,
              int batch_size = 128, bool fill_in_data = false) {
    const int N = W * H * C;

    auto op_ptr = InstantiateOperator(op_spec);

    shared_ptr<Tensor<CPUBackend>> tensor_in(new Tensor<CPUBackend>());
    shared_ptr<Tensor<CPUBackend>> tensor_out(new Tensor<CPUBackend>());
    tensor_in->set_type(TypeInfo::Create<T>());
    tensor_in->Resize({W, H, C});

    if (fill_in_data) {
      auto *ptr = tensor_in->mutable_data<T>();
      for (int i = 0; i < N; i++) {
        ptr[i] = static_cast<T>(i);
      }
    }
    // Create workspace and set input and output
    SampleWorkspace ws;
    ws.AddInput(tensor_in);
    ws.AddOutput(tensor_out);
    ws.set_data_idx(0);
    ws.set_thread_idx(0);
    ws.set_stream(0);

    op_ptr->Run(&ws);
    for (auto _ : st) {
      op_ptr->Run(&ws);
      st.counters["FPS"] = benchmark::Counter(st.iterations() + 1,
        benchmark::Counter::kIsRate);
    }
  }
};

}  // namespace dali

#endif  // DALI_BENCHMARK_OPERATOR_BENCH_H_
