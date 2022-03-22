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
#include "dali/core/format.h"
#include "dali/pipeline/operator/eager_operator.h"
#include "dali/pipeline/util/thread_pool.h"

namespace dali {

/**
 * @brief Debug mode pipeline keeping operators, thread pool and CUDA stream.
 */
class DLL_PUBLIC PipelineDebug {
 public:
  DLL_PUBLIC inline PipelineDebug(int max_batch_size, int num_threads, int device_id,
                                  bool set_affinity = false)
      : max_batch_size(max_batch_size),
        device_id(device_id),
        num_threads(num_threads),
        thread_pool(num_threads, device_id, set_affinity) {
    if (device_id != CPU_ONLY_DEVICE_ID) {
      DeviceGuard g(device_id);
      cuda_stream = CUDAStreamPool::instance().Get(device_id);
    }
  }

  DLL_PUBLIC void AddOperator(OpSpec &spec, int logical_id);

  template <typename InBackend, typename OutBackend>
  DLL_PUBLIC std::vector<std::shared_ptr<TensorList<OutBackend>>> RunOperator(
      int logical_id, const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
      const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
    DALI_FAIL("Unsupported backends in PipelineDebug.RunOperator().");
  }

 private:
  int max_batch_size;
  int device_id;
  int num_threads;
  CUDAStreamLease cuda_stream;
  ThreadPool thread_pool;
  std::unordered_map<int, EagerOperator<CPUBackend>> cpu_operators;
  std::unordered_map<int, EagerOperator<GPUBackend>> gpu_operators;
  std::unordered_map<int, EagerOperator<MixedBackend>> mixed_operators;
};

void PipelineDebug::AddOperator(OpSpec &spec, int logical_id) {
  spec.AddArg("max_batch_size", max_batch_size);
  spec.AddArg("device_id", device_id);
  spec.AddArg("num_threads", num_threads);

  std::string device = spec.GetArgument<std::string>("device");

  if (device == "gpu") {
    gpu_operators.insert({logical_id, EagerOperator<GPUBackend>(spec)});
  } else if (device == "cpu") {
    cpu_operators.insert({logical_id, EagerOperator<CPUBackend>(spec)});
  } else if (device == "mixed") {
    mixed_operators.insert({logical_id, EagerOperator<MixedBackend>(spec)});
  }
}

template <>
std::vector<std::shared_ptr<TensorList<CPUBackend>>> PipelineDebug::RunOperator(
    int logical_id, const std::vector<std::shared_ptr<TensorList<CPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
  auto op = cpu_operators.find(logical_id);
  DALI_ENFORCE(op != cpu_operators.end(), "Failed to acquire CPU Operator in PipelineDebug.");
  return op->second.template Run<CPUBackend, CPUBackend>(inputs, kwargs, &thread_pool);
}

template <>
std::vector<std::shared_ptr<TensorList<GPUBackend>>> PipelineDebug::RunOperator(
    int logical_id, const std::vector<std::shared_ptr<TensorList<GPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
  auto op = gpu_operators.find(logical_id);
  DALI_ENFORCE(op != gpu_operators.end(), "Failed to acquire GPU Operator in PipelineDebug.");
  return op->second.template Run<GPUBackend, GPUBackend>(inputs, kwargs, cuda_stream);
}

template <>
std::vector<std::shared_ptr<TensorList<GPUBackend>>> PipelineDebug::RunOperator(
    int logical_id, const std::vector<std::shared_ptr<TensorList<CPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
  auto op = mixed_operators.find(logical_id);
  DALI_ENFORCE(op != mixed_operators.end(), "Failed to acquire Mixed Operator in PipelineDebug.");
  return op->second.template Run<CPUBackend, GPUBackend>(inputs, kwargs, cuda_stream);
}

}  // namespace dali

#endif  // DALI_PIPELINE_PIPELINE_DEBUG_H_
