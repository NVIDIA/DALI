// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime_api.h>
#include "dali/pipeline/operator/builtin/copy.h"

namespace dali {

template <>
void Copy<GPUBackend>::RunCopies(DeviceWorkspace &ws) {
  scatter_gather_.Run(ws.stream(), true);
}

DALI_REGISTER_OPERATOR(Copy, Copy<GPUBackend>, GPU);

}  // namespace dali
