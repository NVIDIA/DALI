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

#ifndef DALI_PIPELINE_OPERATOR_CALLABLE_OPERATOR_H_
#define DALI_PIPELINE_OPERATOR_CALLABLE_OPERATOR_H_

#include "dali/core/common.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {
// template <typename Backend>
// struct backend_to_input {
//   using backend = CPUBackend;
//   using type = TensorVector<CPUBackend>;
// };

// template <>
// struct backend_to_input<GPUBackend> {
//   using backend = GPUBackend;
//   using type = TensorList<GPUBackend>;
// };

// template <typename Backend>
// struct backend_to_output {
//   using backend = GPUBackend;
//   using type = TensorList<GPUBackend>;
// };

// template <>
// struct backend_to_output<CPUBackend> {
//   using backend = CPUBackend;
//   using type = TensorVector<CPUBackend>;
// };

template <typename Backend>
std::shared_ptr<TensorList<Backend>> AsTensorList(std::shared_ptr<TensorList<Backend>> input) {
  return input;
}

template <typename Backend>
std::shared_ptr<TensorList<Backend>> AsTensorList(std::shared_ptr<TensorVector<Backend>> input) {
  // auto tl = input->AsTensorList(false);
  // tl->set_type(input->type());
  return input->AsTensorList();
}

template <typename Backend>
class DLL_PUBLIC CallableOperator {
 public:
  DLL_PUBLIC CallableOperator(const OpSpec &spec) : op_spec(spec) {
    op_spec.AddArg("num_threads", thread_pool->NumThreads());
    op = InstantiateOperator(op_spec);
  }

  template <typename InBackend, typename OutBackend>
  DLL_PUBLIC inline std::vector<std::shared_ptr<TensorList<OutBackend>>> Run(
      const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
      const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
    DALI_FAIL("Unsupported backends in CallableOperator.");
  }

  DLL_PUBLIC inline void SetOpSpec(const OpSpec &spec) {
    op_spec = spec;
  }

  DLL_PUBLIC inline static void SetThreadPool(int num_threads, int device_id, bool set_affinity) {
    thread_pool = std::make_unique<ThreadPool>(num_threads, device_id, set_affinity);
  }

 private:
  template <typename InBackend, typename OutBackend, typename WSInputType, typename WSOutputType>
  std::vector<std::shared_ptr<TensorList<OutBackend>>> RunImpl(
      const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
      const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs);

  OpSpec op_spec;
  workspace_t<Backend> ws;
  std::unique_ptr<OperatorBase> op;

  static std::unique_ptr<ThreadPool> thread_pool;
};

template <>
template <>
std::vector<std::shared_ptr<TensorList<CPUBackend>>> CallableOperator<CPUBackend>::Run(
    const std::vector<std::shared_ptr<TensorList<CPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
  ws.SetThreadPool(thread_pool.get());

  return RunImpl<CPUBackend, CPUBackend, TensorVector<CPUBackend>, TensorVector<CPUBackend>>(
      inputs, kwargs);
}

template <>
template <>
std::vector<std::shared_ptr<TensorList<GPUBackend>>> CallableOperator<GPUBackend>::Run(
    const std::vector<std::shared_ptr<TensorList<GPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
  ws.set_stream(0);  // TODO(ksztenderski): get correct stream
  CUDA_CALL(cudaStreamSynchronize(0));
  return RunImpl<GPUBackend, GPUBackend, TensorList<GPUBackend>, TensorList<GPUBackend>>(inputs,
                                                                                         kwargs);
}

template <>
template <>
std::vector<std::shared_ptr<TensorList<GPUBackend>>> CallableOperator<MixedBackend>::Run(
    const std::vector<std::shared_ptr<TensorList<CPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
  ws.set_stream(0);
  CUDA_CALL(cudaStreamSynchronize(0));
  return RunImpl<CPUBackend, GPUBackend, TensorVector<CPUBackend>, TensorList<GPUBackend>>(inputs,
                                                                                           kwargs);
}

template <typename Backend>
template <typename InBackend, typename OutBackend, typename WSInputType, typename WSOutputType>
std::vector<std::shared_ptr<TensorList<OutBackend>>> CallableOperator<Backend>::RunImpl(
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

  int batch_size = op_spec.GetArgument<int>("max_batch_size");
  std::vector<OutputDesc> output_desc{};
  std::vector<std::shared_ptr<TensorList<OutBackend>>> outputs{};
  size_t num_outputs = op_spec.GetSchema().NumOutput();

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
std::unique_ptr<ThreadPool> CallableOperator<Backend>::thread_pool =
    std::make_unique<ThreadPool>(1, 0, false);

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_CALLABLE_OPERATOR_H_
