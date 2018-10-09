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

#include "dali/pipeline/operators/crop/slice.h"

namespace dali {

DALI_SCHEMA(Slice)
    .DocStr(R"code(Perform a random crop.)code")
    .NumInput(3)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddArg(
        "crop",
        R"code(Size of the cropped image. If only a single value `c` is provided,
 the resulting crop will be square with size `(c,c)`)code",
        DALI_INT_VEC)

    .EnforceInputLayout(DALI_NHWC)
    .AddParent("Crop");

template <>
void Slice<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  Crop<CPUBackend>::RunImpl(ws, idx);
}

template <>
void Slice<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *ws) {
  DALI_ENFORCE(ws->NumInput() == 3, "Expected 3 inputs. Received: " +
                                        std::to_string(ws->NumInput() == 3));

  // TODO flash attributes
  Crop<CPUBackend>::SetupSharedSampleParams(ws);
}

// Register operator
DALI_REGISTER_OPERATOR(Slice, Slice<CPUBackend>, CPU);

}  // namespace dali
