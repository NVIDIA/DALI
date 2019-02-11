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

#include "dali/pipeline/workspace/sample_workspace.h"

namespace dali {

template <>
const Tensor<CPUBackend>& SampleWorkspace::Input(int idx) const {
  return *CPUInput(idx);
}

template <>
const Tensor<GPUBackend>& SampleWorkspace::Input(int idx) const {
  return *GPUInput(idx);
}

template <>
Tensor<CPUBackend>& SampleWorkspace::Output(int idx) {
  return *CPUOutput(idx);
}

template <>
Tensor<GPUBackend>& SampleWorkspace::Output(int idx) {
  return *GPUOutput(idx);
}

}  // namespace dali
