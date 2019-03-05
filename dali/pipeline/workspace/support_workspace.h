// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_WORKSPACE_SUPPORT_WORKSPACE_H_
#define DALI_PIPELINE_WORKSPACE_SUPPORT_WORKSPACE_H_

#include <utility>
#include <vector>
#include <memory>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

template <typename Backend>
using SupportInputType = shared_ptr<Tensor<Backend>>;
template <typename Backend>
using SupportOutputType = shared_ptr<Tensor<Backend>>;

/**
 * @brief SupportWorkspace stores all data that a support operator operates on,
 * including its input and output Tensors, parameter tensors and
 * meta-data about execution.
 */
class DLL_PUBLIC SupportWorkspace : public WorkspaceBase<SupportInputType, SupportOutputType> {
 public:
  using WorkspaceBase<SupportInputType, SupportOutputType>::input_t;
  using WorkspaceBase<SupportInputType, SupportOutputType>::output_t;
  DLL_PUBLIC SupportWorkspace() {}
  DLL_PUBLIC ~SupportWorkspace() override = default;

  /**
   * @brief Returns the input Tensor at index `idx`.
   */
  template <typename Backend>
  DLL_PUBLIC const Tensor<Backend>& Input(int idx) const;

  /**
   * @brief Returns the output Tensor at index `idx`.
   */
  template <typename Backend>
  DLL_PUBLIC Tensor<Backend>& Output(int idx);
};

}  // namespace dali

#endif  // DALI_PIPELINE_WORKSPACE_SUPPORT_WORKSPACE_H_
