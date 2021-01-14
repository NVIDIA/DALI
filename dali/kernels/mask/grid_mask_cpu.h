// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_MASK_GRID_MASK_CPU_H_
#define DALI_KERNELS_MASK_GRID_MASK_CPU_H_

#include <utility>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

template<typename Type>
class GridMaskCpu {
 public:
  KernelRequirements Setup(KernelContext &context, const TensorShape<> &shape) {
    KernelRequirements req;
    req.output_shapes = { TensorListShape<>{{shape}} };
    return req;
  }

  void Run(KernelContext &context, const OutTensorCPU<Type> &out,
           const InTensorCPU<Type> &in, int tile, float ratio, float angle,
           float sx, float sy) {
    auto in_ptr = in.data;
    auto out_ptr = out.data;
    float ca = cos(angle) / tile;
    float sa = sin(angle) / tile;
    sx /= tile;
    sy /= tile;

    for (int y = 0; y < in.shape[0]; y++) {
      float fxy = -sx + y * -sa;
      float fyy = -sy + y * ca;
      for (int x = 0; x < in.shape[1]; x++) {
        float fx = fxy + x * ca;
        float fy = fyy + x * sa;
        auto m = (fx - floor(fx) >= ratio) || (fy - floor(fy) >= ratio);
        for (int c = 0; c < in.shape[2]; c++)
          *out_ptr++ = *in_ptr++ * m;
      }
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_MASK_GRID_MASK_CPU_H_
