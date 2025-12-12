// Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/generic/shapes.h"

namespace dali {

DALI_SCHEMA(Shapes)
    .DocStr(R"(Returns the shapes of tensors in the input batch.)")
    .NumInput(1)
    .InputDevice(0, InputDevice::Metadata)
    .NumOutput(1)
    .AllowSequences()
    .SupportVolumetric()
    .AddOptionalTypeArg("dtype", "Data type to which the sizes are converted.", DALI_INT64)
    .DeprecateArgInFavorOf("type", "dtype", "0.27")
    .Deprecate("1.44", "", "Use :meth:`nvidia.dali.pipeline.DataNode.shape` instead.");

DALI_SCHEMA(_Shape)
    .DocStr(R"(Returns the shapes of tensors in the input batch.

INTERNAL ONLY; used by DataNode.shape()
)")
    .NumInput(1)
    .InputDevice(0, InputDevice::Metadata)
    .NumOutput(1)
    .AllowSequences()
    .SupportVolumetric()
    .MakeDocHidden()
    .AddOptionalTypeArg("dtype", "Data type to which the sizes are converted.", DALI_INT64);

DALI_REGISTER_OPERATOR(Shapes, Shapes<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Shapes, Shapes<GPUBackend>, GPU);
DALI_REGISTER_OPERATOR(_Shape, Shapes<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(_Shape, Shapes<GPUBackend>, GPU);

}  // namespace dali
