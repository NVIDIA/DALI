// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/static_switch.h"
#include "dali/kernels/imgproc/convolution/separable_convolution_gpu.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/operators/image/convolution/gaussian_blur.h"
#include "dali/operators/image/convolution/gaussian_blur_params.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"
#include "dali/operators/image/convolution/gaussian_blur_gpu.h"

namespace dali {
namespace gaussian_blur {


template op_impl_uptr GetGaussianBlurGpuImpl<int8_t, int8_t>(const OpSpec*, DimDesc);
template op_impl_uptr GetGaussianBlurGpuImpl<float, int8_t>(const OpSpec*, DimDesc);

}  // namespace gaussian_blur
}  // namespace dali
