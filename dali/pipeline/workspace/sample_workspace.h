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

#ifndef DALI_PIPELINE_WORKSPACE_SAMPLE_WORKSPACE_H_
#define DALI_PIPELINE_WORKSPACE_SAMPLE_WORKSPACE_H_

#include <cuda_runtime_api.h>

#include <utility>
#include <vector>
#include <memory>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_list.h"

namespace dali {

/**
 * @brief SampleWorkspace is workspace used for the legacy, deprcated CPU Operator implementation.
 * It has views of all data required for an operator to perform its computation on a single sample,
 * the data is actually owned by a corresponding Workspace
 */
class DLL_PUBLIC SampleWorkspace : public WorkspaceBase<Tensor, std::add_pointer_t> {
 public:
  using Base = WorkspaceBase<Tensor, std::add_pointer_t>;
  DLL_PUBLIC SampleWorkspace() : data_idx_(-1), thread_idx_(-1) {}

  DLL_PUBLIC ~SampleWorkspace() override = default;

  /**
   * @brief Clears the contents of the workspaces, reseting it
   * to a default state.
   */
  DLL_PUBLIC inline void Clear() {
    Base::Clear();
    data_idx_ = -1;
    thread_idx_ = -1;
  }

  int GetInputBatchSize(int) const {
    DALI_FAIL(
        "Impossible function: "
        "Sample workspace is not aware, that there exists such thing as a batch");
  }

  int GetRequestedBatchSize(int) const {
    DALI_FAIL(
        "Impossible function: "
        "Sample workspace is not aware, that there exists such thing as a batch");
  }

  /**
   * @brief Returns the index of the sample that this workspace stores
   * in the input/output batch.
   */
  DLL_PUBLIC inline int data_idx() const override {
    return data_idx_;
  }

  /**
   * @brief Sets the data index for the workspace.
   */
  DLL_PUBLIC inline void set_data_idx(int data_idx) {
    DALI_ENFORCE(data_idx >= 0, "Negative data index not supported.");
    data_idx_ = data_idx;
  }

  /**
   * @brief Returns the index of the thread that will process this data.
   */
  DLL_PUBLIC inline int thread_idx() const override {
    return thread_idx_;
  }

  /**
   * @brief Sets the thread index for the workspace.
   */
  DLL_PUBLIC inline void set_thread_idx(int thread_idx) {
    DALI_ENFORCE(thread_idx >= 0, "Negative thread index not supported.");
    thread_idx_ = thread_idx;
  }

 private:
  int data_idx_, thread_idx_;
};

/**
 * @brief Fill the `sample` with data references to the ones owned by the `batch` for given
 * `data_idx` and set the `thread_idx`.
 */
DLL_PUBLIC void MakeSampleView(SampleWorkspace &sample, Workspace &batch,
                               int data_idx, int thread_idx);

/**
 * @brief Update the TensorList properties based on the ones that were set in the individual
 * samples during execution of the sample-wise operator.
 *
 * After running sample-wise operator we need to fix the Tensor Vector guarantees that were
 * broken by the legacy operators operating just on samples.
 *
 * TODO(klecki): Introduce RAII wrapper for MakeSampleView and FixBatchPropertiesConsistency
 * @param ws The workspace to update after executing samplewise operator
 * @param contiguous If the operator infers outputs and thus uses contiguous allocations
 */
DLL_PUBLIC void FixBatchPropertiesConsistency(Workspace &ws, bool contiguous);

}  // namespace dali

#endif  // DALI_PIPELINE_WORKSPACE_SAMPLE_WORKSPACE_H_
