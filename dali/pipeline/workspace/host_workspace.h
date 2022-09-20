// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_WORKSPACE_HOST_WORKSPACE_H_
#define DALI_PIPELINE_WORKSPACE_HOST_WORKSPACE_H_

#include <utility>
#include <vector>
#include <memory>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

template <typename Backend>
using HostInputType = shared_ptr<TensorList<Backend>>;
template <typename Backend>
using HostOutputType = shared_ptr<TensorList<Backend>>;

class SampleWorkspace;

/**
 * @brief HostWorkspace stores all data that a cpu op operates on.
 * The input data and the output data are per-sample (i.e. stored as vectors of cpu `Tensor`).
 */
class DLL_PUBLIC HostWorkspace : public WorkspaceBase<HostInputType, HostOutputType> {
 public:
  DLL_PUBLIC inline HostWorkspace() {}
  DLL_PUBLIC inline ~HostWorkspace() override = default;

  DLL_PUBLIC inline void SetThreadPool(ThreadPool *pool) {
    thread_pool_ = pool;
  }

  DLL_PUBLIC inline bool HasThreadPool() const {
    return thread_pool_ != nullptr;
  }

  DLL_PUBLIC inline ThreadPool &GetThreadPool() const {
    DALI_ENFORCE(HasThreadPool(), "Workspace does not have a Thread Pool.");
    return *thread_pool_;
  }

  bool has_stream() const override {
    return false;
  };

 private:
  cudaStream_t stream_impl() const override {
    return nullptr;
  }

  ThreadPool* thread_pool_ = nullptr;
};

}  // namespace dali

#endif  // DALI_PIPELINE_WORKSPACE_HOST_WORKSPACE_H_
