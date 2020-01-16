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


#include "dali/operators/image/remap/jitter.h"
#include "dali/operators/util/randomizer_impl_cpu.h"
#include "dali/operators/image/remap/displacement_filter_impl_cpu.h"

namespace dali {

// TODO(ptredak): re-enable it once RNG is changed on the CPU to be deterministic
// DALI_REGISTER_OPERATOR(Jitter, Jitter<CPUBackend>, CPU);

DALI_SCHEMA(Jitter)
    .DocStr(R"code(Perform a random Jitter augmentation.
The output image is produced by moving each pixel by a
random amount bounded by half of `nDegree` parameter
(in both x and y dimensions).)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("nDegree",
        R"code(Each pixel is moved by a random amount in range `[-nDegree/2, nDegree/2]`.)code",
        2)
    .InputLayout(0, "HWC")
    .AddParent("DisplacementFilter");

}  // namespace dali
