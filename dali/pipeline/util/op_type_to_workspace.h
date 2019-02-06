// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_UTIL_OP_TYPE_TO_WORKSPACE_H_
#define DALI_PIPELINE_UTIL_OP_TYPE_TO_WORKSPACE_H_

#include "dali/common.h"
#include "dali/pipeline/workspace/device_workspace.h"
#include "dali/pipeline/workspace/sample_workspace.h"
#include "dali/pipeline/workspace/mixed_workspace.h"
#include "dali/pipeline/workspace/support_workspace.h"

namespace dali {

/**
 * @brief Utility class, which maps the DALIOpType enum to appropriate workspace
 */
template <DALIOpType op_type>
struct workspace_type;

template <>
struct workspace_type<DALIOpType::CPU> {
  using type = HostWorkspace;
};

template <>
struct workspace_type<DALIOpType::GPU> {
  using type = DeviceWorkspace;
};

template <>
struct workspace_type<DALIOpType::MIXED> {
  using type = MixedWorkspace;
};

template <>
struct workspace_type<DALIOpType::SUPPORT> {
  using type = SupportWorkspace;
};

template <DALIOpType op_type>
using workspace_t = typename workspace_type<op_type>::type;

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_OP_TYPE_TO_WORKSPACE_H_
