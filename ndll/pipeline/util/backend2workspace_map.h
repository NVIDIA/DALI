// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_UTIL_BACKEND2WORKSPACE_MAP_H_
#define NDLL_PIPELINE_UTIL_BACKEND2WORKSPACE_MAP_H_

#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/workspace/device_workspace.h"
#include "ndll/pipeline/workspace/sample_workspace.h"

namespace ndll {

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

// Workspace<CPUBackend> maps to SampleWorkspace
// Workspace<GPUBackend> maps to DeviceWorkspace
template<typename Backend>
using Workspace = typename Backend2WorkspaceMap<Backend>::Type;

}  // namespace ndll

#endif  // NDLL_PIPELINE_UTIL_BACKEND2WORKSPACE_MAP_H_
