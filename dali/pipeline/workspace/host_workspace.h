// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
 * HostWorkspace differs from BatchWorkspace in that the input data
 * in a mixed workspace is per-sample, and the outputs are contiguous.
 */
class HostWorkspace : public WorkspaceBase<HostInputType, HostOutputType> {
 public:
  inline HostWorkspace() {}
  inline ~HostWorkspace() = default;

  /**
   * @brief Returns a sample workspace for the given sample
   * index and thread index
   */
  void GetSample(SampleWorkspace *ws, int data_idx, int thread_idx);

  /**
   * @brief Returns the number of Tensors in the input set of
   * tensors at the given index.
   */
  int NumInputAtIdx(int idx) const;

  /**
   * @brief Returns the number of Tensors in the output set of
   * tensors at the given index.
   */
  int NumOutputAtIdx(int idx) const;

  /**
   * @brief Returns the Tensor at index `data_idx` in the input
   * Tensors at index `idx`.
   *
   * @throws runtime_error if the calling type does not match the
   * type of the tensor at the given index
   */
  template <typename Backend>
  const Tensor<Backend>& Input(int idx, int data_idx) const;

  /**
   * @brief Returns the Tensor at index `data_idx` in the output
   * Tensors at index `idx`.
   *
   * @throws runtime_error if the calling type does not match the
   * type of the tensor at the given index
   */
  template <typename Backend>
  Tensor<Backend>* Output(int idx, int data_idx);
};

}  // namespace dali

#endif  // DALI_PIPELINE_WORKSPACE_HOST_WORKSPACE_H_
