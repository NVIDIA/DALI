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
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

template <typename Backend>
class DLL_PUBLIC DirectOperator {
 public:
  DLL_PUBLIC inline DirectOperator(const OpSpec &spec)
      : batch_size(spec.GetArgument<int>("max_batch_size")),
        num_outputs(spec.GetSchema().NumOutput()),
        op(InstantiateOperator(spec)) {}

  template <typename InBackend, typename OutBackend>
  DLL_PUBLIC inline std::vector<std::shared_ptr<TensorList<OutBackend>>> Run(
      const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
      const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
    DALI_FAIL("Unsupported backends in DirectOperator.");
  }

  template <typename InBackend, typename OutBackend>
  DLL_PUBLIC inline std::vector<std::shared_ptr<TensorList<OutBackend>>> Run(
      const std::vector<std::shared_ptr<TensorList<InBackend>>> &inputs,
      const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs,
      ThreadPool *tp);

  DLL_PUBLIC inline static void SetThreadPool(int num_threads, int device_id, bool set_affinity) {
    thread_pool = std::make_unique<ThreadPool>(num_threads, device_id, set_affinity);
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

  static std::unique_ptr<ThreadPool> thread_pool;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_DIRECT_OPERATOR_H_
