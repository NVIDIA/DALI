// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
class SupportWorkspace : public WorkspaceBase<SupportInputType, SupportOutputType> {
 public:
  SupportWorkspace() {}
  ~SupportWorkspace() = default;

  /**
   * @brief Returns the input Tensor at index `idx`.
   */
  const Tensor<CPUBackend>& Input(int idx) const;

  /**
   * @brief Returns the output Tensor at index `idx`.
   */
  Tensor<CPUBackend>* Output(int idx);
};

}  // namespace dali

#endif  // DALI_PIPELINE_WORKSPACE_SUPPORT_WORKSPACE_H_
