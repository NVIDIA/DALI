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

#include <cuda_runtime_api.h>
#include "dali/pipeline/operator/builtin/copy.h"

namespace dali {

template<>
void Copy<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  output.set_type(input.type());
  output.SetLayout(input.GetLayout());
  output.ResizeLike(input);
  CUDA_CALL(cudaMemcpyAsync(
          output.raw_mutable_data(),
          input.raw_data(),
          input.nbytes(),
          cudaMemcpyDeviceToDevice,
          ws.stream()));
}

DALI_REGISTER_OPERATOR(Copy, Copy<GPUBackend>, GPU);

}  // namespace dali
