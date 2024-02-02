// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/image/crop/random_crop_generator.h"

namespace dali {

DALI_SCHEMA(RandomCropGenerator)
  .DocStr(R"code(Produces a cropping window with a randomly selected area and aspect ratio.

Expects a one-dimensional input representing the shape of the input we want to crop (HW or HWC representation).)code")
  .NumInput(1)
  .NumOutput(2)
  .AddParent("RandomCropAttr");

DALI_REGISTER_OPERATOR(RandomCropGenerator, RandomCropGeneratorOp<CPUBackend>, CPU);

}  // namespace dali
