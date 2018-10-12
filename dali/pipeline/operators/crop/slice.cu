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

template <>
void Slice<GPUBackend>::DataDependentSetup(DeviceWorkspace *ws) {}

template <>
void Slice<GPUBackend>::ThreadDependentSetup(DeviceWorkspace *ws) {}

template<>
void Slice<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx)
{
    Crop<GPUBackend>::RunImpl(ws, idx);
}

template<>
void Slice<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace *ws)
{
    DALI_ENFORCE(ws->NumInput() == 3, "Expected 3 inputs. Received: " + std::to_string(ws->NumInput() == 3));

    // TODO flash attributes

    Crop<GPUBackend>::SetupSharedSampleParams(ws);
}


// Register operator
DALI_REGISTER_OPERATOR(Slice, Slice<GPUBackend>, GPU);

}  // namespace dali
