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
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dali/core/cuda_stream_pool.h"
#include "dali/core/nvtx.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/util/batch_utils.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

template <typename Backend>
std::shared_ptr<TensorList<Backend>> AsTensorList(std::shared_ptr<TensorList<Backend>> in) {
  return in;
}

template <typename Backend>
std::shared_ptr<TensorList<Backend>> AsTensorList(std::shared_ptr<TensorVector<Backend>> in) {
  if (in->IsContiguous()) {
    // Filled contiguous TensorVector, we can return TensorList directly.
    auto tl = in->AsTensorList(false);
    // Explicitly set layout (it could be empty in case of per-sample operators).
    tl->SetLayout(in->GetLayout());
    return tl;
  }

  auto tl = std::make_shared<TensorList<Backend>>();
  tl->Copy(*in);
  return tl;
}

template <typename StorageType>
void MakeContiguous(std::shared_ptr<StorageType> storage) {}

template <>
void MakeContiguous(std::shared_ptr<TensorVector<CPUBackend>> storage) {
  storage->SetContiguous(true);
}

template <typename Backend>
struct Backend2Types {};

template <>
struct Backend2Types<CPUBackend> {
  using InBackend = CPUBackend;
  using OutBackend = CPUBackend;
  using WSInputType = TensorVector<CPUBackend>;
  using WSOutputType = TensorVector<CPUBackend>;
  static const char name[8];
};

template <>
struct Backend2Types<GPUBackend> {
  using InBackend = GPUBackend;
  using OutBackend = GPUBackend;
  using WSInputType = TensorList<GPUBackend>;
  using WSOutputType = TensorList<GPUBackend>;
  static const char name[8];
};

template <>
struct Backend2Types<MixedBackend> {
  using InBackend = CPUBackend;
  using OutBackend = GPUBackend;
  using WSInputType = TensorVector<CPUBackend>;
  using WSOutputType = TensorList<GPUBackend>;
  static const char name[8];
};

const char Backend2Types<CPUBackend>::name[] = "CPU";
const char Backend2Types<GPUBackend>::name[] = "GPU";
const char Backend2Types<MixedBackend>::name[] = "Mixed";

/**
 * @brief Direct operator providing eager execution of an operator in Run.
 */
template <typename Backend>
class DLL_PUBLIC EagerOperator {
  using InBackend = typename Backend2Types<Backend>::InBackend;
  using OutBackend = typename Backend2Types<Backend>::OutBackend;
  using WSInputType = typename Backend2Types<Backend>::WSInputType;
  using WSOutputType = typename Backend2Types<Backend>::WSOutputType;

 public:
  DLL_PUBLIC inline EagerOperator(const OpSpec &spec) : EagerOperator(spec, spec.name()) {}

  DLL_PUBLIC inline EagerOperator(const OpSpec &spec, std::string name)
      : EagerOperator(spec, std::move(name), GetSharedThreadPool()->NumThreads()) {}

  DLL_PUBLIC inline EagerOperator(const OpSpec &spec, std::string name, int num_threads)
      : max_batch_size_(spec.GetArgument<int>("max_batch_size")),
        op_spec_(spec),
        name_(std::move(name)) {
    op_spec_.AddArg("num_threads", num_threads);
    op_ = InstantiateOperator(op_spec_);
    num_outputs_ = op_spec_.GetSchema().CalculateOutputs(op_spec_) +
                   op_spec_.GetSchema().CalculateAdditionalOutputs(op_spec_);
  }

  // Runs operator using shared thread pool and shared CUDA stream.
  DLL_PUBLIC std::vector<std::shared_ptr<TensorList<OutBackend>>> Run(
      const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
      const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
      int batch_size = -1);

  // Runs operator using specified thread pool.
  DLL_PUBLIC std::vector<std::shared_ptr<TensorList<OutBackend>>> Run(
      const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
      const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
      ThreadPool *tp, int batch_size = -1);

  // Runs operator using specified CUDA stream.
  DLL_PUBLIC std::vector<std::shared_ptr<TensorList<OutBackend>>> Run(
      const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
      const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
      CUDAStreamLease &cuda_stream, int batch_size = -1);

  DLL_PUBLIC ReaderMeta GetReaderMeta() const {
    ReaderMeta meta = op_->GetReaderMeta();
    DALI_ENFORCE(meta, "Operator " + name_ + " does not expose valid metadata.");
    return meta;
  }

  // Update shared thread pool used for all direct operators.
  DLL_PUBLIC inline static void UpdateThreadPool(int num_threads) {
    std::lock_guard lock(shared_thread_pool_mutex_);

    SharedThreadPoolInstance().reset(
        new ThreadPool(num_threads, CPU_ONLY_DEVICE_ID, false, "EagerOperator"));
  }

  // Update shared CUDA stream used for all direct operators.
  DLL_PUBLIC inline static void UpdateCudaStream(int device_id) {
    if (device_id != CPU_ONLY_DEVICE_ID) {
      DeviceGuard g(device_id);
      CUDAStreamLease &cuda_stream = GetSharedCudaStream();
      cuda_stream = CUDAStreamPool::instance().Get(device_id);
    }
  }

 private:
  std::vector<std::shared_ptr<TensorList<OutBackend>>> RunImpl(
      const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
      const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
      int batch_size = -1);

  inline std::string ExtendErrorMsg(const std::string &backend, const char *what) {
    return make_string("Error when executing ", backend, " operator ", op_spec_.name(),
                       ", instance name: \"", name_, "\", encountered:\n", what);
  }

  static inline int GetDefaultNumThreads() {
    int num_cores = std::thread::hardware_concurrency();
    return num_cores < 6 ? num_cores : 6;
  }

  static inline std::shared_ptr<ThreadPool> &SharedThreadPoolInstance() {
    static std::shared_ptr<ThreadPool> thread_pool = std::make_shared<ThreadPool>(
        GetDefaultNumThreads(), CPU_ONLY_DEVICE_ID, false, "EagerOperator");

    return thread_pool;
  }

  static inline std::shared_ptr<ThreadPool> GetSharedThreadPool() {
    std::shared_lock lock(shared_thread_pool_mutex_);

    return SharedThreadPoolInstance();
  }

  static inline CUDAStreamLease &GetSharedCudaStream() {
    static CUDAStreamLease cuda_stream;

    return cuda_stream;
  }

  int max_batch_size_;
  size_t num_outputs_;
  workspace_t<Backend> ws_;
  OpSpec op_spec_;
  std::string name_;
  std::unique_ptr<OperatorBase> op_;

  static std::shared_mutex shared_thread_pool_mutex_;
};

template <typename Backend>
std::vector<std::shared_ptr<TensorList<typename EagerOperator<Backend>::OutBackend>>>
EagerOperator<Backend>::Run(
    const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
    int batch_size) {
  return Run(inputs, kwargs, GetSharedCudaStream(), batch_size);
}

template <>
std::vector<std::shared_ptr<TensorList<CPUBackend>>> EagerOperator<CPUBackend>::Run(
    const std::vector<std::shared_ptr<TensorList<CPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
    int batch_size) {
  return Run(inputs, kwargs, GetSharedThreadPool().get(), batch_size);
}

template <typename Backend>
std::vector<std::shared_ptr<TensorList<typename EagerOperator<Backend>::OutBackend>>>
EagerOperator<Backend>::Run(
    const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
    ThreadPool *thread_pool, int batch_size) {
  try {
    DomainTimeRange tr("[DALI][" + std::string(Backend2Types<Backend>::name) + " op] " + name_,
                       DomainTimeRange::kBlue1);
    ws_.Clear();
    ws_.SetThreadPool(thread_pool);

    return RunImpl(inputs, kwargs, batch_size);
  } catch (std::exception &e) {
    throw std::runtime_error(ExtendErrorMsg(Backend2Types<Backend>::name, e.what()));
  }
}

template <typename Backend>
std::vector<std::shared_ptr<TensorList<typename EagerOperator<Backend>::OutBackend>>>
EagerOperator<Backend>::Run(
    const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
    CUDAStreamLease &cuda_stream, int batch_size) {
  try {
    DomainTimeRange tr("[DALI][" + std::string(Backend2Types<Backend>::name) + " op] " + name_,
                       DomainTimeRange::knvGreen);
    ws_.Clear();
    ws_.set_stream(cuda_stream);
    auto output = RunImpl(inputs, kwargs, batch_size);
    CUDA_CALL(cudaStreamSynchronize(cuda_stream));
    return output;
  } catch (std::exception &e) {
    throw std::runtime_error(ExtendErrorMsg(Backend2Types<Backend>::name, e.what()));
  }
}

template <typename Backend>
std::vector<std::shared_ptr<TensorList<typename EagerOperator<Backend>::OutBackend>>>
EagerOperator<Backend>::RunImpl(
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
  std::vector<std::shared_ptr<TensorList<OutBackend>>> outputs(num_outputs_);

  for (size_t i = 0; i < num_outputs_; ++i) {
    auto tensor_out = std::make_shared<WSOutputType>(batch_size);
    MakeContiguous(tensor_out);
    ws_.AddOutput(tensor_out);
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
    outputs[i] = AsTensorList<OutBackend>(ws_.template OutputPtr<OutBackend>(i));
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
std::shared_mutex EagerOperator<Backend>::shared_thread_pool_mutex_{};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_EAGER_OPERATOR_H_
