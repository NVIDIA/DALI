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

#include "dali/pipeline/workspace/device_workspace.h"

#include "dali/pipeline/workspace/sample_workspace.h"

namespace dali {

template <>
const TensorList<CPUBackend>& DeviceWorkspace::Input(int idx) const {
  return *CPUInput(idx);
}

template <>
const TensorList<GPUBackend>& DeviceWorkspace::Input(int idx) const {
  return *GPUInput(idx);
}

template <>
TensorList<CPUBackend>& DeviceWorkspace::MutableInput(int idx) {
  return *CPUInput(idx);
}

template <>
TensorList<GPUBackend>& DeviceWorkspace::MutableInput(int idx) {
  return *GPUInput(idx);
}

template <>
TensorList<CPUBackend>& DeviceWorkspace::Output(int idx) {
  return *CPUOutput(idx);
}

template <>
TensorList<GPUBackend>& DeviceWorkspace::Output(int idx) {
  return *GPUOutput(idx);
}

}  // namespace dali
