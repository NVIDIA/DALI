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

#ifndef DALI_PIPELINE_PIPELINE_DEBUG_H_
#define DALI_PIPELINE_PIPELINE_DEBUG_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "dali/core/cuda_stream_pool.h"
#include "dali/pipeline/operator/eager_operator.h"
#include "dali/pipeline/util/thread_pool.h"

namespace dali {

/**
 * @brief Debug mode pipeline keeping operators, thread pool and CUDA stream.
 */
class DLL_PUBLIC PipelineDebug {
  template <typename Backend>
  using input_t = std::shared_ptr<TensorList<typename Backend2Types<Backend>::InBackend>>;

  template <typename Backend>
  using output_t = std::shared_ptr<TensorList<typename Backend2Types<Backend>::OutBackend>>;

 public:
  DLL_PUBLIC inline PipelineDebug(int max_batch_size, int num_threads, int device_id,
                                  bool set_affinity = false)
      : max_batch_size_(max_batch_size),
        device_id_(device_id),
        num_threads_(num_threads),
        thread_pool_(num_threads, device_id, set_affinity, "Debug pipeline") {
    if (device_id != CPU_ONLY_DEVICE_ID) {
      DeviceGuard g(device_id);
      cuda_stream_ = CUDAStreamPool::instance().Get(device_id);
    }
  }

  DLL_PUBLIC void AddOperator(OpSpec &spec, int logical_id) {
    FillOpSpec(spec);
    AddOperatorImpl(spec, logical_id);
  }

  /**
   * @brief Adds many identical operators (used for input sets).
   */
  DLL_PUBLIC void AddMultipleOperators(OpSpec &spec, std::vector<int> &logical_ids) {
    FillOpSpec(spec);
    for (int logical_id : logical_ids) {
      AddOperatorImpl(spec, logical_id);
    }
  }

  template <typename Backend>
  DLL_PUBLIC std::vector<output_t<Backend>> RunOperator(
      int logical_id, const std::vector<input_t<Backend>> &inputs,
      const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
      int batch_size = -1) {
    DALI_FAIL("Unsupported backends in PipelineDebug.RunOperator().");
  }

 private:
  void FillOpSpec(OpSpec &spec) {
    spec.AddArg("max_batch_size", max_batch_size_);
    spec.AddArg("device_id", device_id_);
  }

  void AddOperatorImpl(OpSpec &spec, int logical_id) {
    std::string name = "__debug__" + spec.name() + "_" + std::to_string(logical_id);
    std::string device = spec.GetArgument<std::string>("device");

    if (device == "gpu") {
      gpu_operators_.insert({logical_id, EagerOperator<GPUBackend>(spec, name, num_threads_)});
    } else if (device == "cpu") {
      cpu_operators_.insert({logical_id, EagerOperator<CPUBackend>(spec, name, num_threads_)});
    } else if (device == "mixed") {
      mixed_operators_.insert({logical_id, EagerOperator<MixedBackend>(spec, name, num_threads_)});
    }
  }

  int max_batch_size_;
  int device_id_;
  int num_threads_;
  CUDAStreamLease cuda_stream_;
  ThreadPool thread_pool_;
  std::unordered_map<int, EagerOperator<CPUBackend>> cpu_operators_;
  std::unordered_map<int, EagerOperator<GPUBackend>> gpu_operators_;
  std::unordered_map<int, EagerOperator<MixedBackend>> mixed_operators_;
};

template <>
std::vector<std::shared_ptr<TensorList<CPUBackend>>> PipelineDebug::RunOperator<CPUBackend>(
    int logical_id, const std::vector<std::shared_ptr<TensorList<CPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
    int batch_size) {
  auto op = cpu_operators_.find(logical_id);
  DALI_ENFORCE(op != cpu_operators_.end(), "Failed to acquire CPU Operator in PipelineDebug.");
  return op->second.Run(inputs, kwargs, &thread_pool_, batch_size);
}

template <>
std::vector<std::shared_ptr<TensorList<GPUBackend>>> PipelineDebug::RunOperator<GPUBackend>(
    int logical_id, const std::vector<std::shared_ptr<TensorList<GPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
    int batch_size) {
  auto op = gpu_operators_.find(logical_id);
  DALI_ENFORCE(op != gpu_operators_.end(), "Failed to acquire GPU Operator in PipelineDebug.");
  return op->second.Run(inputs, kwargs, cuda_stream_, batch_size);
}

template <>
std::vector<std::shared_ptr<TensorList<GPUBackend>>> PipelineDebug::RunOperator<MixedBackend>(
    int logical_id, const std::vector<std::shared_ptr<TensorList<CPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
    int batch_size) {
  auto op = mixed_operators_.find(logical_id);
  DALI_ENFORCE(op != mixed_operators_.end(), "Failed to acquire Mixed Operator in PipelineDebug.");
  return op->second.Run(inputs, kwargs, cuda_stream_, batch_size);
}

}  // namespace dali

#endif  // DALI_PIPELINE_PIPELINE_DEBUG_H_
