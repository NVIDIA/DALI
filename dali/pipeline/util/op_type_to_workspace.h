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
 * @brief Utility class, which maps the OpType enum to appropriate workspace
 */
template <OpType op_type>
struct op_type_to_workspace_type;

template <>
struct op_type_to_workspace_type<OpType::CPU> {
  using type = HostWorkspace;
};

template <>
struct op_type_to_workspace_type<OpType::GPU> {
  using type = DeviceWorkspace;
};

template <>
struct op_type_to_workspace_type<OpType::MIXED> {
  using type = MixedWorkspace;
};

template <>
struct op_type_to_workspace_type<OpType::SUPPORT> {
  using type = SupportWorkspace;
};

template <OpType op_type>
using op_type_to_workspace_t = typename op_type_to_workspace_type<op_type>::type;

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_OP_TYPE_TO_WORKSPACE_H_
