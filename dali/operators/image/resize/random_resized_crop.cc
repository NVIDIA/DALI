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
  .DocStr("Perform a crop with randomly chosen area and aspect ratio,"
" then resize it to given size. Expects a 3-dimensional input with samples"
" in HWC layout `(height, width, channels)`.")
  .NumInput(1)
  .NumOutput(1)
  .AddArg("size",
      R"code(Size of resized image.)code",
      DALI_INT_VEC)
  .AddParent("RandomCropAttr")
  .AddParent("ResamplingFilterAttr")
  .InputLayout(0, { "HWC", "CHW" });

template<>
void RandomResizedCrop<CPUBackend>::BackendInit() {
  InitializeCPU(num_threads_);
}

template<>
void RandomResizedCrop<CPUBackend>::RunImpl(HostWorkspace &ws) {
  auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);

  DALI_ENFORCE(input.shape().sample_dim() == 3, "Operator expects 3-dimensional image input.");

  RunResize(ws, output, input);
  output.SetLayout(input.GetLayout());
}

DALI_REGISTER_OPERATOR(RandomResizedCrop, RandomResizedCrop<CPUBackend>, CPU);

}  // namespace dali
