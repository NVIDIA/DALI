// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_WORKSPACE_SAMPLE_WORKSPACE_H_
#define NDLL_PIPELINE_WORKSPACE_SAMPLE_WORKSPACE_H_

#include <cuda_runtime_api.h>

#include <utility>
#include <vector>
#include <memory>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/workspace/device_workspace.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/data/tensor_list.h"
#include "ndll/pipeline/workspace/host_workspace.h"
#include "ndll/pipeline/workspace/mixed_workspace.h"

namespace ndll {

template <typename Backend>
using SampleInputType = shared_ptr<Tensor<Backend>>;
template <typename Backend>
using SampleOutputType = shared_ptr<Tensor<Backend>>;

/**
 * @brief SampleWorkspace stores all data required for an operator to
 * perform its computation on a single sample.
 */
class SampleWorkspace : public Workspace<SampleInputType, SampleOutputType> {
 public:
  SampleWorkspace() : data_idx_(-1), thread_idx_(-1), has_stream_(false) {}

  ~SampleWorkspace() = default;

  /**
   * @brief Clears the contents of the workspaces, reseting it
   * to a default state.
   */
  inline void Clear() {
    data_idx_ = -1;
    thread_idx_ = -1;
    has_stream_ = false;
    stream_ = 0;

    cpu_inputs_.clear();
    gpu_inputs_.clear();
    cpu_outputs_.clear();
    gpu_outputs_.clear();
    input_index_map_.clear();
    output_index_map_.clear();
  }

  /**
   * @brief Returns Tensor with index = data_idx() from the input
   * TensorList at index = `idx`.
   */
  template <typename Backend>
  const Tensor<Backend>& Input(int idx) const;

  /**
   * @brief Returns Tensor with index = data_idx() from the output
   * TensorList at index = `idx`.
   */
  template <typename Backend>
  Tensor<Backend>* Output(int idx);

  /**
   * @brief Returns the index of the sample that this workspace stores
   * in the input/output batch.
   */
  inline int data_idx() const { return data_idx_; }

  /**
   * @brief Sets the data index for the workspace.
   */
  inline void set_data_idx(int data_idx) {
    NDLL_ENFORCE(data_idx >= 0, "Negative data index not supported.");
    data_idx_ = data_idx;
  }

  /**
   * @brief Returns the index of the thread that will process this data.
   */
  inline int thread_idx() const { return thread_idx_; }

  /**
   * @brief Sets the thread index for the workspace.
   */
  inline void set_thread_idx(int thread_idx) {
    NDLL_ENFORCE(thread_idx >= 0, "Negative thread index not supported.");
    thread_idx_ = thread_idx;
  }

  /**
   * @brief Returns true if the workspace contains a valid stream.
   */
  inline bool has_stream() const { return has_stream_; }

  /**
   * @brief Returns the cuda stream that this work is to be done in.
   */
  inline cudaStream_t stream() const {
    NDLL_ENFORCE(has_stream_, "Workspace does not have a valid stream.");
    return stream_;
  }

  /**
   * @brief Sets the stream for this workspace.
   */
  inline void set_stream(cudaStream_t stream) {
    has_stream_ = true;
    stream_ = stream;
  }

 private:
  int data_idx_, thread_idx_;
  cudaStream_t stream_;
  bool has_stream_;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_WORKSPACE_SAMPLE_WORKSPACE_H_
