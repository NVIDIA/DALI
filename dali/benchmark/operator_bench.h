// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

class OperatorBench : public DALIBenchmark {
 public:
  template <typename OutputContainer, typename OperatorPtr, typename Workspace>
  void Setup(OperatorPtr &op_ptr, const OpSpec &spec, Workspace &ws, int batch_size) {
    std::vector<OutputDesc> outputs;
    bool can_infer_outs = op_ptr->CanInferOutputs();
    if (op_ptr->Setup(outputs, ws) && can_infer_outs) {
      int num_out = outputs.size();
      for (int i = 0; i < num_out; i++) {
        auto data_out = std::make_shared<OutputContainer>(batch_size);
        data_out->Resize(outputs[i].shape, outputs[i].type);
        ws.AddOutput(data_out);
      }
    } else {
      for (int i = 0; i < spec.GetSchema().NumOutput(); i++) {
        ws.AddOutput(std::make_shared<OutputContainer>(batch_size));
      }
    }
  }

  template <typename T>
  void RunCPU(benchmark::State& st, const OpSpec &op_spec,
              int batch_size = 128,
              int H = 1080, int W = 1920, int C = 3,
              bool fill_in_data = false, int num_threads = 4) {
    const int N = W * H * C;

    auto op_ptr = InstantiateOperator(op_spec);

    auto data_in = std::make_shared<TensorList<CPUBackend>>(batch_size);
    data_in->set_type<T>();
    data_in->Resize(uniform_list_shape(batch_size, TensorShape<>{H, W, C}));
    data_in->SetLayout("HWC");

    if (fill_in_data) {
      for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
        auto *ptr = data_in->template mutable_tensor<T>(sample_idx);
        for (int i = 0; i < N; i++) {
          ptr[i] = static_cast<T>(i);
        }
      }
    }
    // Create workspace and set input and output
    Workspace ws;
    ws.AddInput(data_in);
    ThreadPool tp(num_threads, 0, false, "OperatorBench");
    ws.SetThreadPool(&tp);

    Setup<TensorList<CPUBackend>>(op_ptr, op_spec, ws, batch_size);
    op_ptr->Run(ws);
    for (auto _ : st) {
      op_ptr->Run(ws);
      st.counters["FPS"] = benchmark::Counter(st.iterations() + 1,
        benchmark::Counter::kIsRate);
    }
  }

  template <typename T>
  void RunGPU(benchmark::State &st, const OpSpec &op_spec, int batch_size = 128,
              TensorListShape<> shape = uniform_list_shape(128, {1080, 1920, 3}),
              TensorLayout layout = "HWC",
              bool fill_in_data = false,
              int64_t sync_each_n = -1) {
    assert(layout.size() == shape.sample_dim());

    auto op_ptr = InstantiateOperator(op_spec);

    auto data_in_cpu = std::make_shared<TensorList<CPUBackend>>();
    data_in_cpu->set_type<T>();
    data_in_cpu->Resize(shape);
    data_in_cpu->SetLayout(layout);
    if (fill_in_data) {
      for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
        auto *ptr = data_in_cpu->template mutable_tensor<T>(sample_idx);
        for (int i = 0; i < volume(shape[sample_idx]); i++) {
            ptr[i] = static_cast<T>(i);
        }
      }
    }
    auto data_in_gpu = std::make_shared<TensorList<GPUBackend>>();
    data_in_gpu->Copy(*data_in_cpu, (cudaStream_t)0);
    CUDA_CALL(cudaStreamSynchronize(0));

    // Create workspace and set input and output
    Workspace ws;
    ws.AddInput(data_in_gpu);
    ws.set_stream(0);

    Setup<TensorList<GPUBackend>>(op_ptr, op_spec, ws, batch_size);
    op_ptr->Run(ws);
    CUDA_CALL(cudaStreamSynchronize(0));

    int64_t batches = 0;

    while (st.KeepRunning()) {
      op_ptr->Run(ws);
      batches++;

      if (batches == st.max_iterations || (sync_each_n > 0 && batches % sync_each_n == 0)) {
        CUDA_CALL(cudaStreamSynchronize(0));
      }
    }
    st.counters["FPS"] = benchmark::Counter(batch_size * st.iterations(),
                                            benchmark::Counter::kIsRate);
  }

  template <typename T>
  void RunGPU(benchmark::State &st, const OpSpec &op_spec, int batch_size = 128,
              TensorShape<> shape = {1080, 1920, 3}, TensorLayout layout = "HWC",
              bool fill_in_data = false, int64_t sync_each_n = -1) {
    RunGPU<T>(st, op_spec, batch_size,
              uniform_list_shape(batch_size, shape), layout, fill_in_data, sync_each_n);
  }

  template <typename T>
  void RunGPU(benchmark::State& st, const OpSpec &op_spec,
              int batch_size = 128, int H = 1080, int W = 1920, int C = 3,
              bool fill_in_data = false, int64_t sync_each_n = -1) {
    RunGPU<T>(st, op_spec, batch_size, {H, W, C}, "HWC", fill_in_data, sync_each_n);
  }
};

}  // namespace dali

#endif  // DALI_BENCHMARK_OPERATOR_BENCH_H_
