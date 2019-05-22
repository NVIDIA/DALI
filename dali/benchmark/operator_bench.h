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
#include <memory>
#include <vector>
#include "dali/benchmark/dali_bench.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

class OperatorBench : public DALIBenchmark {
 public:
  template <typename T>
  void RunCPU(benchmark::State& st, OpSpec op_spec,
              int batch_size = 128,
              int H = 1080, int W = 1920, int C = 3,
              bool fill_in_data = false) {
    const int N = W * H * C;

    auto op_ptr = InstantiateOperator(op_spec);

    shared_ptr<Tensor<CPUBackend>> data_in(new Tensor<CPUBackend>());
    shared_ptr<Tensor<CPUBackend>> data_out(new Tensor<CPUBackend>());
    data_in->set_type(TypeInfo::Create<T>());
    data_in->Resize({W, H, C});

    if (fill_in_data) {
      auto *ptr = data_in->mutable_data<T>();
      for (int i = 0; i < N; i++) {
        ptr[i] = static_cast<T>(i);
      }
    }
    // Create workspace and set input and output
    SampleWorkspace ws;
    ws.AddInput(data_in);
    ws.AddOutput(data_out);
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

  template <typename T>
  void RunGPU(benchmark::State& st, OpSpec op_spec,
              int batch_size = 128,
              int H = 1080, int W = 1920, int C = 3,
              bool fill_in_data = false) {
    const int N = W * H * C;

    auto op_ptr = InstantiateOperator(op_spec);

    auto data_in_cpu = std::make_shared<TensorList<CPUBackend>>();
    data_in_cpu->set_type(TypeInfo::Create<T>());
    data_in_cpu->Resize(std::vector<Dims>(batch_size, {W, H, C}));
    if (fill_in_data) {
      for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
        auto *ptr = data_in_cpu->mutable_tensor<T>(sample_idx);
        for (int i = 0; i < N; i++) {
            ptr[i] = static_cast<T>(i);
        }
      }
    }
    auto data_in_gpu = std::make_shared<TensorList<GPUBackend>>();
    data_in_gpu->Copy(*data_in_cpu, (cudaStream_t)0);
    CUDA_CALL(cudaStreamSynchronize(0));

    auto data_out_gpu = std::make_shared<TensorList<GPUBackend>>();

    // Create workspace and set input and output
    DeviceWorkspace ws;
    ws.AddInput(data_in_gpu);
    ws.AddOutput(data_out_gpu);
    ws.set_stream(0);

    op_ptr->Run(&ws);
    CUDA_CALL(cudaStreamSynchronize(0));
    for (auto _ : st) {
      op_ptr->Run(&ws);
      CUDA_CALL(cudaStreamSynchronize(0));

      int num_batches = st.iterations() + 1;
      st.counters["FPS"] = benchmark::Counter(batch_size * num_batches,
        benchmark::Counter::kIsRate);
    }
  }
};

}  // namespace dali

#endif  // DALI_BENCHMARK_OPERATOR_BENCH_H_
