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

#include "dali/pipeline/operator/direct_operator.h"

namespace dali {
template <>
template <>
inline std::vector<std::shared_ptr<TensorList<CPUBackend>>> DirectOperator<CPUBackend>::Run(
    const std::vector<std::shared_ptr<TensorList<CPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
  ws.SetThreadPool(thread_pool.get());

  return RunImpl<CPUBackend, CPUBackend, TensorVector<CPUBackend>, TensorVector<CPUBackend>>(
      inputs, kwargs);
}

template <>
template <>
inline std::vector<std::shared_ptr<TensorList<GPUBackend>>> DirectOperator<GPUBackend>::Run(
    const std::vector<std::shared_ptr<TensorList<GPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
  ws.set_stream(0);  // TODO(ksztenderski): get correct stream
  CUDA_CALL(cudaStreamSynchronize(0));
  return RunImpl<GPUBackend, GPUBackend, TensorList<GPUBackend>, TensorList<GPUBackend>>(inputs,
                                                                                         kwargs);
}

template <>
template <>
inline std::vector<std::shared_ptr<TensorList<GPUBackend>>> DirectOperator<MixedBackend>::Run(
    const std::vector<std::shared_ptr<TensorList<CPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
  ws.set_stream(0);
  CUDA_CALL(cudaStreamSynchronize(0));
  return RunImpl<CPUBackend, GPUBackend, TensorVector<CPUBackend>, TensorList<GPUBackend>>(inputs,
                                                                                           kwargs);
}

template <>
template <>
inline std::vector<std::shared_ptr<TensorList<CPUBackend>>> DirectOperator<CPUBackend>::Run(
    const std::vector<std::shared_ptr<TensorList<CPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
    ThreadPool *tp) {
  ws.SetThreadPool(tp);

  return RunImpl<CPUBackend, CPUBackend, TensorVector<CPUBackend>, TensorVector<CPUBackend>>(
      inputs, kwargs);
}

template <typename Backend>
template <typename InBackend, typename OutBackend, typename WSInputType, typename WSOutputType>
std::vector<std::shared_ptr<TensorList<OutBackend>>> DirectOperator<Backend>::RunImpl(
    const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
  ws.Clear();

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

  // Setup outputs. We need to SetBatchSizes first, so we cannot do it in the previous loop.
  if (op->Setup(output_desc, ws) && op->CanInferOutputs()) {
    for (size_t i = 0; i < num_outputs; ++i) {
      ws.template Output<OutBackend>(i).Resize(output_desc[i].shape, output_desc[i].type);
    }
  }

  op->Run(ws);

  for (size_t i = 0; i < num_outputs; ++i) {
    // TODO(ksztenderski): Remove Copy
    auto tl = std::make_shared<TensorList<OutBackend>>();
    tl->Copy(ws.template Output<OutBackend>(i));
    outputs.push_back(tl);
  }

  return outputs;
}

template <typename Backend>
std::unique_ptr<ThreadPool> DirectOperator<Backend>::thread_pool =
    std::make_unique<ThreadPool>(1, 0, false);

}  // namespace dali