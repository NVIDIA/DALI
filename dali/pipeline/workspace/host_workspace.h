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

#ifndef DALI_PIPELINE_WORKSPACE_HOST_WORKSPACE_H_
#define DALI_PIPELINE_WORKSPACE_HOST_WORKSPACE_H_

#include <utility>
#include <vector>
#include <memory>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

template <typename Backend>
using HostInputType = vector<shared_ptr<Tensor<Backend>>>;
template <typename Backend>
using HostOutputType = vector<shared_ptr<Tensor<Backend>>>;

class SampleWorkspace;

/**
 * @brief HostWorkspace stores all data that a cpu op operates on.
 * The input data and the output data are per-sample (i.e. stored as vectors of cpu `Tensor`).
 */
class DLL_PUBLIC HostWorkspace : public WorkspaceBase<HostInputType, HostOutputType> {
 public:
  using WorkspaceBase<HostInputType, HostOutputType>::input_t;
  using WorkspaceBase<HostInputType, HostOutputType>::output_t;

  DLL_PUBLIC inline HostWorkspace() {}
  DLL_PUBLIC inline ~HostWorkspace() override = default;

  /**
   * @brief Returns a sample workspace for the given sample
   * index and thread index
   */
  DLL_PUBLIC void GetSample(SampleWorkspace *ws, int data_idx, int thread_idx);

  /**
   * @brief Returns the number of Tensors in the input set of
   * tensors at the given index.
   */
  DLL_PUBLIC int NumInputAtIdx(int idx) const;

  /**
   * @brief Returns the number of Tensors in the output set of
   * tensors at the given index.
   */
  DLL_PUBLIC int NumOutputAtIdx(int idx) const;

  /**
   * @brief Returns the Tensor at index `data_idx` in the input
   * Tensors at index `idx`.
   *
   * @throws runtime_error if the calling type does not match the
   * type of the tensor at the given index
   */
  template <typename Backend>
  DLL_PUBLIC const Tensor<Backend>& Input(int idx, int data_idx) const;

  /**
   * @brief Returns the Tensor at index `data_idx` in the output
   * Tensors at index `idx`.
   *
   * @throws runtime_error if the calling type does not match the
   * type of the tensor at the given index
   */
  template <typename Backend>
  DLL_PUBLIC Tensor<Backend>& Output(int idx, int data_idx);
};

}  // namespace dali

#endif  // DALI_PIPELINE_WORKSPACE_HOST_WORKSPACE_H_
