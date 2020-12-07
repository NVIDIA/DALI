// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_UTIL_OPERATOR_IMPL_UTILS_H_
#define DALI_PIPELINE_UTIL_OPERATOR_IMPL_UTILS_H_

#include <string>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/tensor_view.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/workspace/device_workspace.h"
#include "dali/pipeline/workspace/sample_workspace.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

/**
 * @brief Utility interface to be used as a base for argument/type specific operator implementations
 */
template <typename Backend>
class OpImplBase {
 public:
  virtual ~OpImplBase() = default;
  virtual bool SetupImpl(std::vector<OutputDesc> &output_desc,
                         const workspace_t<Backend> &ws) = 0;
  virtual void RunImpl(workspace_t<Backend> &ws) = 0;
};

template <>
class OpImplBase<CPUBackend> {
 public:
  virtual ~OpImplBase() = default;
  virtual bool SetupImpl(std::vector<OutputDesc> &output_desc,
                         const workspace_t<CPUBackend> &ws) = 0;
  virtual void RunImpl(HostWorkspace &ws) {
    assert(false);
  }
  virtual void RunImpl(SampleWorkspace &ws) {
    assert(false);
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_OPERATOR_IMPL_UTILS_H_
