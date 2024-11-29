// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include "dali/operators/image/remap/jitter.cuh"
#include "dali/operators/image/remap/displacement_filter_impl_gpu.cuh"

namespace dali {

DALI_SCHEMA(Jitter)
  .DocStr(R"code(Performs a random Jitter augmentation.

The output images are produced by moving each pixel by a random amount, in the x and y dimensions,
and bounded by half of the `nDegree` parameter.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("nDegree",
      R"code(Each pixel is moved by a random amount in the ``[-nDegree/2, nDegree/2]`` range)code",
      2)
  .InputLayout(0, "HWC")
  .AddRandomSeedArg()
  .AddParent("DisplacementFilter");

DALI_REGISTER_OPERATOR(Jitter, Jitter<GPUBackend>, GPU);

}  // namespace dali
