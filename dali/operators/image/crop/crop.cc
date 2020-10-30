// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/image/crop/crop.h"
#include <tuple>
#include <vector>
#include "dali/core/static_switch.h"
#include "dali/kernels/slice/slice_cpu.h"
#include "dali/pipeline/data/views.h"
#include "dali/util/half.hpp"

namespace dali {

DALI_SCHEMA(Crop)
    .DocStr(R"code(Crops the images with the specified window dimensions and window position
(upper left corner).)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowSequences()
    .SupportVolumetric()
    .AddOptionalArg<DALIImageType>("image_type", "Image type", nullptr)
    .DeprecateArg("image_type")  // deprecated since 0.24dev
    .AddParent("CropAttr")
    .AddParent("OutOfBoundsAttr")
    .AddParent("SliceBase");

// Register operator
DALI_REGISTER_OPERATOR(Crop, Crop<CPUBackend>, CPU);

}  // namespace dali
