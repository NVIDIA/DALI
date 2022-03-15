// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_DIRECT_OPERATOR_H_
#define DALI_PIPELINE_OPERATOR_DIRECT_OPERATOR_H_

#include "dali/core/common.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

template <typename Backend>
std::shared_ptr<TensorList<Backend>> AsTensorList(std::shared_ptr<TensorList<Backend>> input) {
  return input;
}

template <typename Backend>
std::shared_ptr<TensorList<Backend>> AsTensorList(std::shared_ptr<TensorVector<Backend>> input) {
  auto tl = std::make_shared<TensorList<Backend>>();
  tl->Copy(*input);
  return tl;
}

/**
 * @brief Direct operator providing direct execution of an operator in Run.
 */
template <typename Backend>
class DLL_PUBLIC DirectOperator {
 public:
  DLL_PUBLIC inline DirectOperator(const OpSpec &spec)
      : batch_size(spec.GetArgument<int>("max_batch_size")),
        num_outputs(spec.GetSchema().NumOutput()),
        op(InstantiateOperator(spec)) {}

  // Runs operator using shared thread pool and shared CUDA stream.
  template <typename InBackend, typename OutBackend>
  DLL_PUBLIC std::vector<std::shared_ptr<TensorList<OutBackend>>> Run(
      const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
      const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
    DALI_FAIL("Unsupported backends in DirectOperator.Run().");
  }

  // Runs operator using specified thread pool.
  template <typename InBackend, typename OutBackend>
  DLL_PUBLIC std::vector<std::shared_ptr<TensorList<OutBackend>>> Run(
      const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
      const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
      ThreadPool *tp) {
    DALI_FAIL("Unsupported backends in DirectOperator.Run() with thread pool.");
  }

  // Runs operator using specified CUDA stream.
  template <typename InBackend, typename OutBackend>
  DLL_PUBLIC std::vector<std::shared_ptr<TensorList<OutBackend>>> Run(
      const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
      const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
      cudaStream_t cuda_stream) {
    DALI_FAIL("Unsupported backends in DirectOperator.Run() with CUDA stream");
  }

  DLL_PUBLIC inline static void SetThreadPool(int num_threads, int device_id, bool set_affinity) {
    shared_thread_pool = std::make_unique<ThreadPool>(num_threads, device_id, set_affinity);
  }

  DLL_PUBLIC inline static void SetCudaStream(int device_id) {
    if (device_id != CPU_ONLY_DEVICE_ID) {
      DeviceGuard g(device_id);
      shared_cuda_stream = CUDAStreamPool::instance().Get(device_id);
    }
  }

 private:
  template <typename InBackend, typename OutBackend, typename WSInputType, typename WSOutputType>
  std::vector<std::shared_ptr<TensorList<OutBackend>>> RunImpl(
      const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
      const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs);

  int batch_size;
  size_t num_outputs;
  workspace_t<Backend> ws;
  std::unique_ptr<OperatorBase> op;

  static cudaStream_t shared_cuda_stream;
  static std::unique_ptr<ThreadPool> shared_thread_pool;
};

template <>
template <>
std::vector<std::shared_ptr<TensorList<CPUBackend>>> DirectOperator<CPUBackend>::Run(
    const std::vector<std::shared_ptr<TensorList<CPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
    ThreadPool *thread_pool) {
  ws.SetThreadPool(thread_pool);

  return RunImpl<CPUBackend, CPUBackend, TensorVector<CPUBackend>, TensorVector<CPUBackend>>(
      inputs, kwargs);
}

template <>
template <>
std::vector<std::shared_ptr<TensorList<GPUBackend>>> DirectOperator<GPUBackend>::Run(
    const std::vector<std::shared_ptr<TensorList<GPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
    cudaStream_t cuda_stream) {
  ws.set_stream(cuda_stream);
  CUDA_CALL(cudaStreamSynchronize(cuda_stream));
  auto output = RunImpl<GPUBackend, GPUBackend, TensorList<GPUBackend>, TensorList<GPUBackend>>(
      inputs, kwargs);
  CUDA_CALL(cudaStreamSynchronize(cuda_stream));
  return output;
}

template <>
template <>
std::vector<std::shared_ptr<TensorList<GPUBackend>>> DirectOperator<MixedBackend>::Run(
    const std::vector<std::shared_ptr<TensorList<CPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
    cudaStream_t cuda_stream) {
  ws.set_stream(cuda_stream);
  CUDA_CALL(cudaStreamSynchronize(cuda_stream));
  auto output = RunImpl<CPUBackend, GPUBackend, TensorVector<CPUBackend>, TensorList<GPUBackend>>(
      inputs, kwargs);
  CUDA_CALL(cudaStreamSynchronize(cuda_stream));
  return output;
}

template <>
template <>
std::vector<std::shared_ptr<TensorList<CPUBackend>>> DirectOperator<CPUBackend>::Run(
    const std::vector<std::shared_ptr<TensorList<CPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
  return Run<CPUBackend, CPUBackend>(inputs, kwargs, shared_thread_pool.get());
}

template <>
template <>
std::vector<std::shared_ptr<TensorList<GPUBackend>>> DirectOperator<GPUBackend>::Run(
    const std::vector<std::shared_ptr<TensorList<GPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
  return Run<GPUBackend, GPUBackend>(inputs, kwargs, shared_cuda_stream);
}

template <>
template <>
std::vector<std::shared_ptr<TensorList<GPUBackend>>> DirectOperator<MixedBackend>::Run(
    const std::vector<std::shared_ptr<TensorList<CPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
  return Run<CPUBackend, GPUBackend>(inputs, kwargs, shared_cuda_stream);
}

template <typename Backend>
template <typename InBackend, typename OutBackend, typename WSInputType, typename WSOutputType>
std::vector<std::shared_ptr<TensorList<OutBackend>>> DirectOperator<Backend>::RunImpl(
    const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
  ws.Clear();

  // Converts and adds inputs to the workspace.
  for (auto &input : inputs) {
    auto tensor_in = std::make_shared<WSInputType>();
    tensor_in->ShareData(*input);
    ws.AddInput(tensor_in);
  }

  for (auto &arg : kwargs) {
    ws.AddArgumentInput(arg.first, arg.second);
  }

  std::vector<OutputDesc> output_desc{};
  std::vector<std::shared_ptr<TensorList<OutBackend>>> outputs{};

  outputs.reserve(num_outputs);

  for (size_t i = 0; i < num_outputs; ++i) {
    ws.AddOutput(std::make_shared<WSOutputType>(batch_size));
  }

  ws.SetBatchSizes(batch_size);

  // Setup outputs.
  if (op->Setup(output_desc, ws) && op->CanInferOutputs()) {
    for (size_t i = 0; i < num_outputs; ++i) {
      ws.template Output<OutBackend>(i).Resize(output_desc[i].shape, output_desc[i].type);
    }
  }

  op->Run(ws);

  for (size_t i = 0; i < num_outputs; ++i) {
    // TODO(ksztenderski): Remove Copy for TV.
    outputs.push_back(AsTensorList<OutBackend>(ws.template OutputPtr<OutBackend>(i)));
  }

  return outputs;
}

template <typename Backend>
std::unique_ptr<ThreadPool> DirectOperator<Backend>::shared_thread_pool =
    std::make_unique<ThreadPool>(1, 0, false);

template <typename Backend>
cudaStream_t DirectOperator<Backend>::shared_cuda_stream = 0;

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_DIRECT_OPERATOR_H_
