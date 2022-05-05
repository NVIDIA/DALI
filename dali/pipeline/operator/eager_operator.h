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

#ifndef DALI_PIPELINE_OPERATOR_EAGER_OPERATOR_H_
#define DALI_PIPELINE_OPERATOR_EAGER_OPERATOR_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>


#include "dali/core/cuda_stream_pool.h"
#include "dali/core/nvtx.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/util/batch_utils.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

template <typename Backend>
std::shared_ptr<TensorList<Backend>> AsTensorList(const std::shared_ptr<TensorList<Backend>> &in) {
  return in;
}

template <typename Backend>
std::shared_ptr<TensorList<Backend>> AsTensorList(
    const std::shared_ptr<TensorVector<Backend>> &in) {
  if (in->IsContiguous()) {
    // Filled contiguous TensorVector, we can return TensorList directly.
    return in->AsTensorList(false);
  }

  auto tl = std::make_shared<TensorList<Backend>>();
  tl->Copy(*in);
  return tl;
}

/**
 * @brief Direct operator providing eager execution of an operator in Run.
 */
template <typename Backend>
class DLL_PUBLIC EagerOperator {
 public:
  DLL_PUBLIC inline EagerOperator(const OpSpec &spec) : EagerOperator(spec, spec.name()) {}

  DLL_PUBLIC inline EagerOperator(const OpSpec &spec, std::string name)
      : max_batch_size_(spec.GetArgument<int>("max_batch_size")),
        op_spec_(spec),
        name_(std::move(name)),
        op_(InstantiateOperator(spec)) {
    num_outputs_ = op_spec_.GetSchema().CalculateOutputs(op_spec_) +
                   op_spec_.GetSchema().CalculateAdditionalOutputs(op_spec_);
  }

  // Runs operator using shared thread pool and shared CUDA stream.
  template <typename InBackend, typename OutBackend>
  DLL_PUBLIC std::vector<std::shared_ptr<TensorList<OutBackend>>> Run(
      const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
      const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
      int batch_size = -1) {
    DALI_FAIL("Unsupported backends in EagerOperator.Run().");
  }

  // Runs operator using specified thread pool.
  template <typename InBackend, typename OutBackend>
  DLL_PUBLIC std::vector<std::shared_ptr<TensorList<OutBackend>>> Run(
      const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
      const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
      ThreadPool *tp, int batch_size = -1) {
    DALI_FAIL("Unsupported backends in EagerOperator.Run() with thread pool.");
  }

  // Runs operator using specified CUDA stream.
  template <typename InBackend, typename OutBackend>
  DLL_PUBLIC std::vector<std::shared_ptr<TensorList<OutBackend>>> Run(
      const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
      const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
      CUDAStreamLease &cuda_stream, int batch_size = -1) {
    DALI_FAIL("Unsupported backends in EagerOperator.Run() with CUDA stream");
  }

  // Update shared thread pool used for all direct operators.
  DLL_PUBLIC inline static void UpdateThreadPool(int num_threads, int device_id,
                                                 bool set_affinity) {
    shared_thread_pool = std::make_unique<ThreadPool>(num_threads, device_id, set_affinity);
  }

  // Update shared CUDA stream used for all direct operators.
  DLL_PUBLIC inline static void UpdateCudaStream(int device_id) {
    if (device_id != CPU_ONLY_DEVICE_ID) {
      DeviceGuard g(device_id);
      shared_cuda_stream = CUDAStreamPool::instance().Get(device_id);
    }
  }

 private:
  template <typename InBackend, typename OutBackend, typename WSInputType, typename WSOutputType>
  std::vector<std::shared_ptr<TensorList<OutBackend>>> RunImpl(
      const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
      const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
      int batch_size = -1);

  inline std::string ExtendErrorMsg(const std::string &backend, const char *what) {
    return make_string("Error when executing ", backend, " operator ", op_spec_.name(),
                       ", instance name: \"", name_, "\", encountered:\n", what);
  }

  int max_batch_size_;
  size_t num_outputs_;
  workspace_t<Backend> ws_;
  OpSpec op_spec_;
  std::string name_;
  std::unique_ptr<OperatorBase> op_;

  static CUDAStreamLease shared_cuda_stream;
  static std::unique_ptr<ThreadPool> shared_thread_pool;
};

template <>
template <>
std::vector<std::shared_ptr<TensorList<CPUBackend>>> EagerOperator<CPUBackend>::Run(
    const std::vector<std::shared_ptr<TensorList<CPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
    ThreadPool *thread_pool, int batch_size) {
  try {
    DomainTimeRange tr("[DALI][CPU op] " + name_, DomainTimeRange::kBlue1);
    ws_.Clear();
    ws_.SetThreadPool(thread_pool);

    return RunImpl<CPUBackend, CPUBackend, TensorVector<CPUBackend>, TensorVector<CPUBackend>>(
        inputs, kwargs, batch_size);
  } catch (std::exception &e) { throw std::runtime_error(ExtendErrorMsg("CPU", e.what())); }
}

template <>
template <>
std::vector<std::shared_ptr<TensorList<GPUBackend>>> EagerOperator<GPUBackend>::Run(
    const std::vector<std::shared_ptr<TensorList<GPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
    CUDAStreamLease &cuda_stream, int batch_size) {
  try {
    DomainTimeRange tr("[DALI][GPU op] " + name_, DomainTimeRange::knvGreen);
    ws_.Clear();
    ws_.set_stream(cuda_stream);
    auto output = RunImpl<GPUBackend, GPUBackend, TensorList<GPUBackend>, TensorList<GPUBackend>>(
        inputs, kwargs, batch_size);
    CUDA_CALL(cudaStreamSynchronize(cuda_stream));
    return output;
  } catch (std::exception &e) { throw std::runtime_error(ExtendErrorMsg("GPU", e.what())); }
}

template <>
template <>
std::vector<std::shared_ptr<TensorList<GPUBackend>>> EagerOperator<MixedBackend>::Run(
    const std::vector<std::shared_ptr<TensorList<CPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
    CUDAStreamLease &cuda_stream, int batch_size) {
  try {
    DomainTimeRange tr("[DALI][Mixed op] " + name_, DomainTimeRange::kOrange);
    ws_.Clear();
    ws_.set_stream(cuda_stream);
    auto output = RunImpl<CPUBackend, GPUBackend, TensorVector<CPUBackend>, TensorList<GPUBackend>>(
        inputs, kwargs, batch_size);
    CUDA_CALL(cudaStreamSynchronize(cuda_stream));
    return output;
  } catch (std::exception &e) { throw std::runtime_error(ExtendErrorMsg("Mixed", e.what())); }
}

template <>
template <>
std::vector<std::shared_ptr<TensorList<CPUBackend>>> EagerOperator<CPUBackend>::Run(
    const std::vector<std::shared_ptr<TensorList<CPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
    int batch_size) {
  return Run<CPUBackend, CPUBackend>(inputs, kwargs, shared_thread_pool.get(), batch_size);
}

template <>
template <>
std::vector<std::shared_ptr<TensorList<GPUBackend>>> EagerOperator<GPUBackend>::Run(
    const std::vector<std::shared_ptr<TensorList<GPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
    int batch_size) {
  return Run<GPUBackend, GPUBackend>(inputs, kwargs, shared_cuda_stream, batch_size);
}

template <>
template <>
std::vector<std::shared_ptr<TensorList<GPUBackend>>> EagerOperator<MixedBackend>::Run(
    const std::vector<std::shared_ptr<TensorList<CPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
    int batch_size) {
  return Run<CPUBackend, GPUBackend>(inputs, kwargs, shared_cuda_stream, batch_size);
}

template <typename Backend>
template <typename InBackend, typename OutBackend, typename WSInputType, typename WSOutputType>
std::vector<std::shared_ptr<TensorList<OutBackend>>> EagerOperator<Backend>::RunImpl(
    const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
    int batch_size) {
  DALI_ENFORCE(batch_size <= max_batch_size_,
               make_string("Expected batch size lower or equal to max batch size. Requested: ",
                           batch_size, " > ", max_batch_size_));
  // Convert and add inputs to the workspace.
  for (size_t in_idx = 0; in_idx < inputs.size(); ++in_idx) {
    auto tensor_in = std::make_shared<WSInputType>();
    tensor_in->ShareData(*inputs[in_idx]);
    int cur_batch_size = tensor_in->num_samples();

    if (batch_size == -1) {
      batch_size = cur_batch_size;
    }

    DALI_ENFORCE(cur_batch_size == batch_size,
                 make_string("Expected uniform batch size in a single operator. Expected: ",
                             batch_size, ", input ", in_idx, " batch size: ", cur_batch_size));
    DALI_ENFORCE(
        cur_batch_size <= max_batch_size_,
        make_string("Expected batch size lower or equal to max batch size. Expected at most: ",
                    max_batch_size_, ", input ", in_idx, " batch size: ", batch_size));

    SetDefaultLayoutIfNeeded(*tensor_in, op_spec_.GetSchema(), in_idx);
    ws_.AddInput(tensor_in);
  }

  if (batch_size == -1) {
    batch_size = max_batch_size_;
  }

  for (auto &arg : kwargs) {
    ws_.AddArgumentInput(arg.first, arg.second);
  }

  std::vector<OutputDesc> output_desc{};
  std::vector<std::shared_ptr<TensorList<OutBackend>>> outputs{};

  outputs.reserve(num_outputs_);

  for (size_t i = 0; i < num_outputs_; ++i) {
    ws_.AddOutput(std::make_shared<WSOutputType>(max_batch_size_));
  }

  ws_.SetBatchSizes(batch_size);

  // Setup outputs.
  if (op_->Setup(output_desc, ws_) && op_->CanInferOutputs()) {
    for (size_t i = 0; i < num_outputs_; ++i) {
      ws_.template Output<OutBackend>(i).Resize(output_desc[i].shape, output_desc[i].type);
    }
  }

  op_->Run(ws_);

  for (size_t i = 0; i < num_outputs_; ++i) {
    outputs.push_back(AsTensorList<OutBackend>(ws_.template OutputPtr<OutBackend>(i)));
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    int cur_batch_size = outputs[i]->num_samples();
    DALI_ENFORCE(cur_batch_size == batch_size,
                 make_string("Unexpected batch size for output ", i, ". Expected: ", batch_size,
                             ", returned: ", cur_batch_size));
  }

  return outputs;
}

template <typename Backend>
std::unique_ptr<ThreadPool> EagerOperator<Backend>::shared_thread_pool =
    std::make_unique<ThreadPool>(1, CPU_ONLY_DEVICE_ID, false);

template <typename Backend>
CUDAStreamLease EagerOperator<Backend>::shared_cuda_stream{};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_EAGER_OPERATOR_H_
