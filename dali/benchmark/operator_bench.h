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
        data_out->Resize(outputs[i].shape);
        data_out->set_type(outputs[i].type);
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

    auto data_in = std::make_shared<TensorVector<CPUBackend>>(batch_size);
    for (auto &in_ptr : *data_in) {
      in_ptr = std::make_shared<Tensor<CPUBackend>>();
      in_ptr->set_type(TypeInfo::Create<T>());
      in_ptr->Resize({H, W, C});
      in_ptr->SetLayout("HWC");
    }

    if (fill_in_data) {
      for (auto &in_ptr : *data_in) {
        auto *ptr = in_ptr->template mutable_data<T>();
        for (int i = 0; i < N; i++) {
          ptr[i] = static_cast<T>(i);
        }
      }
    }
    // Create workspace and set input and output
    HostWorkspace ws;
    ws.AddInput(data_in);
    ThreadPool tp(num_threads, 0, false);
    ws.SetThreadPool(&tp);

    Setup<TensorVector<CPUBackend>>(op_ptr, op_spec, ws, batch_size);
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
              bool fill_in_data = false) {
    assert(layout.size() == shape.size());

    auto op_ptr = InstantiateOperator(op_spec);

    auto data_in_cpu = std::make_shared<TensorList<CPUBackend>>();
    data_in_cpu->set_type(TypeInfo::Create<T>());
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
    DeviceWorkspace ws;
    ws.AddInput(data_in_gpu);
    ws.set_stream(0);

    Setup<TensorList<GPUBackend>>(op_ptr, op_spec, ws, batch_size);
    op_ptr->Run(ws);
    CUDA_CALL(cudaStreamSynchronize(0));
    for (auto _ : st) {
      op_ptr->Run(ws);
      CUDA_CALL(cudaStreamSynchronize(0));

      int num_batches = st.iterations() + 1;
      st.counters["FPS"] = benchmark::Counter(batch_size * num_batches,
        benchmark::Counter::kIsRate);
    }
  }

  template <typename T>
  void RunGPU(benchmark::State &st, const OpSpec &op_spec, int batch_size = 128,
              TensorShape<> shape = {1080, 1920, 3}, TensorLayout layout = "HWC",
              bool fill_in_data = false) {
    RunGPU<T>(st, op_spec, batch_size,
              uniform_list_shape(batch_size, shape), layout, fill_in_data);
  }

  template <typename T>
  void RunGPU(benchmark::State& st, const OpSpec &op_spec,
              int batch_size = 128, int H = 1080, int W = 1920, int C = 3,
              bool fill_in_data = false) {
    RunGPU<T>(st, op_spec, batch_size, {H, W, C}, "HWC", fill_in_data);
  }
};

}  // namespace dali

#endif  // DALI_BENCHMARK_OPERATOR_BENCH_H_
