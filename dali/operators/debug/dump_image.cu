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

#include "dali/operators/debug/dump_image.h"
#include "dali/util/image.h"

namespace dali {

template<>
void DumpImage<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);

  WriteHWCBatch(input, suffix_ + "-" + std::to_string(0));

  // Forward the input
  output.Copy(input, ws.stream());
}

DALI_REGISTER_OPERATOR(DumpImage, DumpImage<GPUBackend>, GPU);

}  // namespace dali
