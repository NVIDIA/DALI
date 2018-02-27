// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_MIXED_WORKSPACE_H_
#define NDLL_PIPELINE_MIXED_WORKSPACE_H_

#include <vector>
#include <utility>
#include <memory>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/data/tensor_list.h"
#include "ndll/pipeline/workspace/workspace.h"

namespace ndll {

template <typename Backend>
using MixedInputType = vector<shared_ptr<Tensor<Backend>>>;
template <typename Backend>
using MixedOutputType = shared_ptr<TensorList<Backend>>;

class SampleWorkspace;

/**
 * @brief MixedWorkspace stores all data that an mixed op operates on.
 * MixedWorkspace differs from BatchWorkspace in that the input data
 * in a mixed workspace is per-sample, and the outputs are contiguous.
 */
class MixedWorkspace : public Workspace<MixedInputType, MixedOutputType> {
 public:
  inline MixedWorkspace() : stream_(0) {}
  inline ~MixedWorkspace() = default;


  /**
   * @brief Returns the number of Tensors in the input set of
   * tensors at the given index.
   */
  int NumInputAtIdx(int idx) const;

  /**
   * @brief Returns the input Tensor at index `data_idx` in the input
   * set of Tensors at index `idx`.
   *
   * @throws runtime_error If calling type does not match the type of
   * the output at the given index.
   */
  template <typename Backend>
  const Tensor<Backend>& Input(int idx, int data_idx) const;

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

 private:
  bool has_stream_ = false, has_event_ = false;
  cudaStream_t stream_;
  cudaEvent_t event_;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_MIXED_WORKSPACE_H_
