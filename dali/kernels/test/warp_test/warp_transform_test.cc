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

#include <gtest/gtest.h>
#include "dali/kernels/imgproc/warp/affine.h"
#include "dali/kernels/imgproc/warp/sphere.h"
#include "dali/kernels/imgproc/warp/water.h"
#include "dali/kernels/imgproc/warp/mapping_traits.h"

static_assert(dali::kernels::warp::is_fp_mapping<dali::kernels::AffineMapping<3>>::value,
              "AffineMapping should be considered fp");
