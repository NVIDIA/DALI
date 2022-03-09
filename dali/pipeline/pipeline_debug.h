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

#include "dali/core/common.h"
#include "dali/core/format.h"
#include "dali/pipeline/operator/direct_operator.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/thread_pool.h"

namespace dali {

class DLL_PUBLIC PipelineDebug {
 public:
  DLL_PUBLIC inline PipelineDebug(int max_batch_size, int num_threads, int device_id,
                                  bool set_affinity = false)
      : max_batch_size(max_batch_size),
        device_id(device_id),
        num_threads(num_threads),
        thread_pool(num_threads, device_id, set_affinity) {}

  void DLL_PUBLIC AddOperator(OpSpec &spec, size_t logical_id);

  template <typename InBackend, typename OutBackend>
  DLL_PUBLIC inline std::vector<std::shared_ptr<TensorList<OutBackend>>> RunOperator(
      size_t logical_id, const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
      const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
    DALI_FAIL("Unsupported backends in PipelineDebug.RunOperator().");
  }

 private:
  int max_batch_size;
  int device_id;
  int num_threads;
  std::unordered_map<size_t, DirectOperator<CPUBackend>> cpu_operators;
  std::unordered_map<size_t, DirectOperator<GPUBackend>> gpu_operators;
  std::unordered_map<size_t, DirectOperator<MixedBackend>> mixed_operators;
  ThreadPool thread_pool;
};

}  // namespace dali

#endif  // DALI_PIPELINE_PIPELINE_DEBUG_H_
