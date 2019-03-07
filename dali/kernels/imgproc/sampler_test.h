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

#ifndef DALI_KERNELS_IMGPROC_SAMPLER_TEST_H_
#define DALI_KERNELS_IMGPROC_SAMPLER_TEST_H_

#include <gtest/gtest.h>
#include "dali/kernels/alloc.h"
#include "dali/kernels/imgproc/sampler.h"
#include "dali/kernels/imgproc/surface.h"

namespace dali {
namespace kernels {

struct SamplerTestData {
  Surface2D<const uint8_t> GetSurface(bool gpu) {
    constexpr int W = 4;
    constexpr int H = 5;
    constexpr int C = 3;
    static const uint8_t data[W*H*C] = {
      0x00, 0x00, 0x00,  0xff, 0xff, 0xff,  0x00, 0x00, 0xff,  0xff, 0xff, 0xff,
      0xff, 0xff, 0xff,  0xff, 0x00, 0x00,  0xff, 0xff, 0xff,  0xff, 0xff, 0x00,
      0x00, 0xff, 0x00,  0xff, 0xff, 0xff,  0x00, 0x00, 0x00,  0xff, 0x00, 0xff,
      0xff, 0xff, 0xff,  0x00, 0xff, 0xff,  0xff, 0xff, 0xff,  0xff, 0x00, 0xff,
      0x00, 0x00, 0xff,  0xff, 0xff, 0xff,  0xff, 0xff, 0x00,  0xff, 0xff, 0xff,
    };

    Surface2D<const uint8_t> surface;
    if (gpu) {
      storage = memory::alloc_unique<uint8_t>(AllocType::GPU, W*H*C);
      cudaMemcpy(storage.get(), data, sizeof(data), cudaMemcpyHostToDevice);
      surface.data = storage.get();
    } else {
      surface.data = data;
    }
    surface.width    = W;
    surface.height   = H;
    surface.channels = C;
    surface.pixel_stride   = C;
    surface.row_stride     = W*C;
    surface.channel_stride = 1;
    return surface;
  }

  memory::KernelUniquePtr<uint8_t> storage;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_SAMPLER_TEST_H_
