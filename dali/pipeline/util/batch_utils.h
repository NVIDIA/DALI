// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_UTIL_BATCH_UTILS_H_
#define DALI_PIPELINE_UTIL_BATCH_UTILS_H_

#include <memory>
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

/**
 * @brief Verifies that the inputs in the workspace satisfy the layout
 *        constraints imposed by the schema.
 */
inline void CheckInputLayouts(const Workspace &ws, const OpSpec &spec) {
  auto &schema = spec.GetSchema();
  for (int i = 0; i < spec.NumRegularInput(); ++i) {
    if (ws.InputIsType<CPUBackend>(i)) {
      auto &input = ws.Input<CPUBackend>(i);
      (void)schema.GetInputLayout(i, input.shape().sample_dim(), input.GetLayout());
    } else if (ws.InputIsType<GPUBackend>(i)) {
      auto &input = ws.Input<GPUBackend>(i);
      (void)schema.GetInputLayout(i, input.shape().sample_dim(), input.GetLayout());
    } else {
      DALI_FAIL(make_string("Input ", i, " has an unknown backend"));
    }
  }
}

template <typename InputRef>
bool SetDefaultLayoutIfNeeded(InputRef &in, const OpSchema &schema, int in_idx) {
  if (!in.GetLayout().empty()) {
    return false;
  }
  auto default_layout = schema.GetInputLayout(in_idx, in.shape().sample_dim(), in.GetLayout());
  if (default_layout.empty()) {
    return false;
  }
  in.SetLayout(default_layout);
  return true;
}

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_BATCH_UTILS_H_
