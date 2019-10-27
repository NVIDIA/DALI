// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_UTIL_BACKEND2WORKSPACE_MAP_H_
#define DALI_PIPELINE_UTIL_BACKEND2WORKSPACE_MAP_H_

#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/workspace/device_workspace.h"
#include "dali/pipeline/workspace/sample_workspace.h"
#include "dali/pipeline/workspace/mixed_workspace.h"

namespace dali {

/**
 * @brief Utility class, which maps the type of workspace with
 * proper Backend
 */
template <typename Backend>
struct Backend2WorkspaceMap {};

template <>
struct Backend2WorkspaceMap<CPUBackend> { using type = SampleWorkspace; };

template <>
struct Backend2WorkspaceMap<GPUBackend> { using type = DeviceWorkspace; };

template <>
struct Backend2WorkspaceMap<MixedBackend> { using type = MixedWorkspace; };


// Workspace<CPUBackend> maps to SampleWorkspace
// Workspace<GPUBackend> maps to DeviceWorkspace
// Workspace<MixedBackend> maps to MixedWorkspace
template <typename Backend>
using Workspace = typename Backend2WorkspaceMap<Backend>::type;

// Actual trait-conventions used, maps as above with exception of CPUBackend -> HostWorkspace
template <typename Backend>
struct backend_to_ws {};

template <>
struct backend_to_ws<CPUBackend> { using type = HostWorkspace; };

template <>
struct backend_to_ws<MixedBackend> { using type = MixedWorkspace; };

template <>
struct backend_to_ws<GPUBackend> { using type = DeviceWorkspace; };

template <typename Backend>
using workspace_t = typename backend_to_ws<Backend>::type;

template <OpType>
struct op_to_workspace;

template <>
struct op_to_workspace<OpType::CPU> { using type = HostWorkspace; };

template <>
struct op_to_workspace<OpType::MIXED> { using type = MixedWorkspace; };

template <>
struct op_to_workspace<OpType::GPU> { using type = DeviceWorkspace; };

template <OpType op_type>
using op_to_workspace_t = typename op_to_workspace<op_type>::type;


template <typename T>
struct workspace_to_op;

template <>
struct workspace_to_op<HostWorkspace> { static constexpr OpType value = OpType::CPU; };

template <>
struct workspace_to_op<MixedWorkspace> { static constexpr OpType value = OpType::MIXED; };

template <>
struct workspace_to_op<DeviceWorkspace> { static constexpr OpType value = OpType::GPU; };


}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_BACKEND2WORKSPACE_MAP_H_
