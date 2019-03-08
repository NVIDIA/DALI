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

#ifndef DALI_PIPELINE_WORKSPACE_MIXED_WORKSPACE_H_
#define DALI_PIPELINE_WORKSPACE_MIXED_WORKSPACE_H_

#include <vector>
#include <utility>
#include <memory>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

template <typename Backend>
using MixedInputType = vector<shared_ptr<Tensor<Backend>>>;
template <typename Backend>
using MixedOutputType = shared_ptr<TensorList<Backend>>;

/**
 * @brief MixedWorkspace stores all data that an mixed op operates on.
 * The input data is per-sample (i.e. vectors of cpu `Tensor`),
 * and the output data is contiguous (i.e cpu or gpu `TensorList`).
 */
class DLL_PUBLIC MixedWorkspace : public WorkspaceBase<MixedInputType, MixedOutputType> {
 public:
  using WorkspaceBase<MixedInputType, MixedOutputType>::input_t;
  using WorkspaceBase<MixedInputType, MixedOutputType>::output_t;
  DLL_PUBLIC inline MixedWorkspace() : stream_(0) {}
  DLL_PUBLIC inline ~MixedWorkspace() override = default;


  /**
   * @brief Returns the number of Tensors in the input set of
   * tensors at the given index.
   */
  DLL_PUBLIC int NumInputAtIdx(int idx) const;

  /**
   * @brief Returns the input Tensor at index `data_idx` in the input
   * set of Tensors at index `idx`.
   *
   * @throws runtime_error If calling type does not match the type of
   * the output at the given index.
   */
  template <typename Backend>
  DLL_PUBLIC const Tensor<Backend>& Input(int idx, int data_idx) const;

  /**
   * @brief Returns the output TensorList at index `idx`.
   *
   * @throws runtime_error If calling type does not match the type of
   * the output at the given index.
   */
  template <typename Backend>
  DLL_PUBLIC TensorList<Backend>& Output(int idx);

  /**
   * @brief Sets the stream for this workspace.
   */
  DLL_PUBLIC inline void set_stream(cudaStream_t stream) {
    has_stream_ = true;
    stream_ = stream;
  }

  /**
   * @brief Returns true if 'set_stream' has been called.
   */
  DLL_PUBLIC inline bool has_stream() const { return has_stream_; }

  /**
   * @brief Returns the cuda stream that this work is to be done in.
   */
  DLL_PUBLIC inline cudaStream_t stream() const {
    DALI_ENFORCE(has_stream_, "Workspace does not have a stream.");
    return stream_;
  }

  /**
   * @brief Sets the event for this workspace.
   */
  DLL_PUBLIC inline void set_event(cudaEvent_t event) {
    has_event_ = true;
    event_ = event;
  }

  /**
   * @brief Returns true if 'set_event' has been called.
   */
  DLL_PUBLIC inline bool has_event() const { return has_event_; }

  /**
   * @brief Returns the cuda event that signals this works completion.
   */
  DLL_PUBLIC inline cudaEvent_t event() const {
    DALI_ENFORCE(has_event_, "Workspace does not have an event.");
    return event_;
  }

 private:
  bool has_stream_ = false, has_event_ = false;
  cudaStream_t stream_;
  cudaEvent_t event_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_WORKSPACE_MIXED_WORKSPACE_H_
