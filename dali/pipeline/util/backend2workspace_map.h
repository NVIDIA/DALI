// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_PIPELINE_UTIL_BACKEND2WORKSPACE_MAP_H_
#define DALI_PIPELINE_UTIL_BACKEND2WORKSPACE_MAP_H_

#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/workspace/device_workspace.h"
#include "dali/pipeline/workspace/sample_workspace.h"
#include "dali/pipeline/workspace/mixed_workspace.h"
#include "dali/pipeline/workspace/support_workspace.h"

namespace dali {

/**
 * @brief Utility clas, which maps the type of workspace with
 * proper Backend
 */
template <typename Backend>
class Backend2WorkspaceMap {};

template<>
class Backend2WorkspaceMap<CPUBackend> {
 public:
  typedef SampleWorkspace Type;
};

template<>
class Backend2WorkspaceMap<GPUBackend> {
 public:
  typedef DeviceWorkspace Type;
};

template<>
class Backend2WorkspaceMap<SupportBackend> {
 public:
  typedef SupportWorkspace Type;
};

template<>
class Backend2WorkspaceMap<MixedBackend> {
 public:
  typedef MixedWorkspace Type;
};


// Workspace<CPUBackend> maps to SampleWorkspace
// Workspace<GPUBackend> maps to DeviceWorkspace
// Workspace<MixedBackend> maps to MixedWorkspace
// Workspace<SupportBackend> maps to SupportWorkspace
template<typename Backend>
using Workspace = typename Backend2WorkspaceMap<Backend>::Type;

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_BACKEND2WORKSPACE_MAP_H_
