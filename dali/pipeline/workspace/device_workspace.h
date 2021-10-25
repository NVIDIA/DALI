// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_WORKSPACE_DEVICE_WORKSPACE_H_
#define DALI_PIPELINE_WORKSPACE_DEVICE_WORKSPACE_H_

#include <cuda_runtime_api.h>

#include <utility>
#include <vector>
#include <memory>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

template <typename Backend>
using DeviceInputType = shared_ptr<TensorList<Backend>>;
template <typename Backend>
using DeviceOutputType = shared_ptr<TensorList<Backend>>;

/**
 * @brief DeviceWorkspace stores all data that a gpu operator operates on,
 * including its input and output TensorLists, parameter tensors and
 * meta-data about execution.
 */
class DLL_PUBLIC DeviceWorkspace : public WorkspaceBase<DeviceInputType, DeviceOutputType> {
 public:
  DLL_PUBLIC DeviceWorkspace() : stream_(0), event_(nullptr) {}
  DLL_PUBLIC ~DeviceWorkspace() override = default;

  /**
   * @brief Clears the contents of the workspaces, resetting it
   * to a default state.
   */
  DLL_PUBLIC inline void Clear() {
    WorkspaceBase<DeviceInputType, DeviceOutputType>::Clear();
    has_stream_ = false;
    has_event_ = false;
    stream_ = 0;
    parent_events_.clear();
  }

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
  DLL_PUBLIC inline bool has_stream() const override {
    return has_stream_;
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

  /**
   * @brief Adds a parent event that will signal this
   * work is allowed to execute.
   */
  DLL_PUBLIC inline void AddParentEvent(cudaEvent_t event) { parent_events_.push_back(event); }

  /**
   * @brief Returns the set of parent events this workspace stores.
   */
  DLL_PUBLIC inline vector<cudaEvent_t> ParentEvents() const { return parent_events_; }

 private:
  cudaStream_t stream_impl() const override {
    return stream_;
  }

  bool has_stream_ = false, has_event_ = false;
  cudaStream_t stream_;
  cudaEvent_t event_;
  vector<cudaEvent_t> parent_events_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_WORKSPACE_DEVICE_WORKSPACE_H_
