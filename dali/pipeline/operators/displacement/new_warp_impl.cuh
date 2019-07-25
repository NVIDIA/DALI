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

#ifndef DALI_PIPELINE_OPERATORS_DISPLACEMENT_NEW_WARP_IMPL_CUH_
#define DALI_PIPELINE_OPERATORS_DISPLACEMENT_NEW_WARP_IMPL_CUH_

#include "dali/pipeline/operators/displacement/new_warp.h"
#include "dali/kernels/imgproc/warp_gpu.h"

namespace dali {

template <typename Mapping>
class NewWarp<GPUBackend> : public Operator<GPUBackend> {
 public:
  NewWarp(const OpSpec &spec) : Operator<GPUBackend>(spec) {}

  using Backend = GPUBackend;

  void RunImpl(DeviceWorkspace* ws, const int idx) override;

  virtual void SetupMapping(DeviceWorkspace* ws) = 0;

  bool InferOutputs(
    std::vector<kernels::TensorListShape<>> &shapes,
    std::vector<TypeInfo> &types, DeviceWorkspace &ws) {
    shapes.resize(1);
    types.resize(1);

  }

  void RunImpl(DeviceWorkspace* ws, const int idx) {
    SetupMapping(ws);

    auto input = ws->Input<GPUBackend>(0);

  }

 protected:
  kernels::KernelManager kmgr;

};


}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DISPLACEMENT_NEW_WARP_IMPL_CUH_
