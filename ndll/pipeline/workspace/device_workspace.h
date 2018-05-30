// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_WORKSPACE_DEVICE_WORKSPACE_H_
#define NDLL_PIPELINE_WORKSPACE_DEVICE_WORKSPACE_H_

#include <cuda_runtime_api.h>

#include <utility>
#include <vector>
#include <memory>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/data/tensor_list.h"
#include "ndll/pipeline/workspace/workspace.h"

namespace ndll {

template <typename Backend>
using DeviceInputType = shared_ptr<TensorList<Backend>>;
template <typename Backend>
using DeviceOutputType = shared_ptr<TensorList<Backend>>;

/**
 * @brief DeviceWorkspace stores all data that a gpu operator operates on,
 * including its input and output TensorLists, parameter tensors and
 * meta-data about execution.
 */
class DeviceWorkspace : public WorkspaceBase<DeviceInputType, DeviceOutputType> {
 public:
  DeviceWorkspace() : stream_(0) {}
  ~DeviceWorkspace() = default;

  /**
   * @brief Clears the contents of the workspaces, reseting it
   * to a default state.
   */
  inline void Clear() {
    WorkspaceBase<DeviceInputType, DeviceOutputType>::Clear();
    has_stream_ = false;
    has_event_ = false;
    stream_ = 0;
    parent_events_.clear();
  }

  /**
   * @brief Returns the input TensorList at index `idx`.
   *
   * @throws runtime_error If calling type does not match the type of
   * the output at the given index.
   */
  template <typename Backend>
  const TensorList<Backend>& Input(int idx) const;

  /**
   * @brief Returns the output TensorList at index `idx`.
   *
   * @throws runtime_error If calling type does not match the type of
   * the output at the given index.
   */
  template <typename Backend>
  TensorList<Backend>* Output(int idx);

  /**
   * @brief Sets the stream for this workspace.
   */
  inline void set_stream(cudaStream_t stream) {
    has_stream_ = true;
    stream_ = stream;
  }

  /**
   * @brief Returns true if 'set_stream' has been called.
   */
  inline bool has_stream() const { return has_stream_; }

  /**
   * @brief Returns the cuda stream that this work is to be done in.
   */
  inline cudaStream_t stream() const {
    NDLL_ENFORCE(has_stream_, "Workspace does not have a stream.");
    return stream_;
  }

  /**
   * @brief Sets the event for this workspace.
   */
  inline void set_event(cudaEvent_t event) {
    has_event_ = true;
    event_ = event;
  }

  /**
   * @brief Returns true if 'set_event' has been called.
   */
  inline bool has_event() const { return has_event_; }

  /**
   * @brief Returns the cuda event that signals this works completion.
   */
  inline cudaEvent_t event() const {
    NDLL_ENFORCE(has_event_, "Workspace does not have an event.");
    return event_;
  }

  /**
   * @brief Adds a parent event that will signal this
   * work is allowed to execute.
   */
  inline void AddParentEvent(cudaEvent_t event) { parent_events_.push_back(event); }

  /**
   * @brief Returns the set of parent events this workspace stores.
   */
  inline vector<cudaEvent_t> ParentEvents() const { return parent_events_; }

 private:
  bool has_stream_ = false, has_event_ = false;
  cudaStream_t stream_;
  cudaEvent_t event_;
  vector<cudaEvent_t> parent_events_;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_WORKSPACE_DEVICE_WORKSPACE_H_
