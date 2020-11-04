// Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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


#include <vector>
#include <random>

#include "dali/operators/image/resize/random_resized_crop.h"
#include "dali/pipeline/operator/common.h"
#include "dali/util/ocv.h"

namespace dali {

DALI_SCHEMA(RandomResizedCrop)
  .DocStr(R"code(Performs a crop with a randomly selected area and aspect ratio and
resizes it to the specified size.

Expects a three-dimensional input with samples in height, width, channels (HWC) layout.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddArg("size",
      R"code(Size of the resized image.)code",
      DALI_INT_VEC)
  .AddParent("RandomCropAttr")
  .AddParent("ResamplingFilterAttr")
  .AllowSequences()
  .InputLayout(0, { "HWC", "CHW", "FHWC", "FCHW", "CFHW" });

template<>
void RandomResizedCrop<CPUBackend>::BackendInit() {
  InitializeCPU(num_threads_);
}

template<>
void RandomResizedCrop<CPUBackend>::RunImpl(HostWorkspace &ws) {
  auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);

  RunResize(ws, output, input);
  output.SetLayout(input.GetLayout());
}

DALI_REGISTER_OPERATOR(RandomResizedCrop, RandomResizedCrop<CPUBackend>, CPU);

}  // namespace dali
