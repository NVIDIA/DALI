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

#ifndef DALI_PIPELINE_WORKSPACE_MIXED_WORKSPACE_H_
#define DALI_PIPELINE_WORKSPACE_MIXED_WORKSPACE_H_

#include <vector>
#include <utility>
#include <memory>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

template <typename Backend>
using MixedInputType = shared_ptr<TensorList<Backend>>;
template <typename Backend>
using MixedOutputType = shared_ptr<TensorList<Backend>>;

/**
 * @brief MixedWorkspace stores all data that an mixed op operates on.
 * The input data is per-sample (i.e. vectors of cpu `Tensor`),
 * and the output data is contiguous (i.e cpu or gpu `TensorList`).
 */
class DLL_PUBLIC MixedWorkspace : public WorkspaceBase<MixedInputType, MixedOutputType> {
 public:
  DLL_PUBLIC inline MixedWorkspace() : stream_(0), event_(nullptr) {}
  DLL_PUBLIC inline ~MixedWorkspace() override = default;

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

 private:
  cudaStream_t stream_impl() const override {
    return stream_;
  }

  bool has_stream_ = false, has_event_ = false;
  cudaStream_t stream_;
  cudaEvent_t event_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_WORKSPACE_MIXED_WORKSPACE_H_
