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
#include <vector>
#include "dali/kernels/imgproc/sampler_test.h"

namespace dali {
namespace kernels {

using Pixel = std::array<uint8_t, 3>;

template <typename Out, DALIInterpType interp, int MaxChannels = 8, typename In>
__global__ void RunSampler(
      Surface2D<Out> out, Sampler<interp, In> sampler, In border_value,
      float dx, float dy, float x0, float y0) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= out.width || y >= out.height)
    return;

  In border[3] = { border_value, border_value, border_value };

  float fx = x * dx + x0;
  float fy = y * dy + y0;
  Out tmp[MaxChannels];
  sampler(tmp, fx, fy, border);
  for (int c = 0; c < out.channels; c++)
    out(x, y, c) = tmp[c];
}

TEST(SamplerGPU, NN) {
  SamplerTestData sd;
  auto surf_cpu = sd.GetSurface(false);
  auto surf_gpu = sd.GetSurface(true);

  ASSERT_EQ(surf_cpu.channels, 3);
  ASSERT_EQ(surf_gpu.channels, 3);
  Sampler<DALI_INTERP_NN, uint8_t> sampler(surf_gpu);

  uint8_t border_value = 50;
  Pixel border = { border_value, border_value, border_value };

  float dy = 0.125f;
  float dx = 0.125f;
  float x0 = -1;
  float y0 = -1;

  int h = (surf_cpu.height+2) / dy + 1;
  int w = (surf_cpu.width+2) / dx + 1;
  int c = surf_cpu.channels;

  auto out_mem = memory::alloc_unique<uint8_t>(AllocType::GPU, w*h*c);
  Surface2D<uint8_t> out_surf = { out_mem.get(), w, h, c, c, w*c, 1 };

  std::vector<uint8_t> out_mem_cpu(w*h*c);

  dim3 block(32, 32);
  dim3 grid((w+31)/32, (h+31)/32);
  RunSampler<<<grid, block>>>(out_surf, sampler, border_value, dx, dy, x0, y0);
  Surface2D<uint8_t> out_cpu = { out_mem_cpu.data(), w, h, c, c, w*c, 1 };

  cudaMemcpy(out_cpu.data, out_surf.data, w*h*c, cudaMemcpyDeviceToHost);

  for (int oy = 0; oy < h; oy++) {
    float y = oy * dy + y0;
    int iy = floorf(y);
    for (int ox = 0; ox < w; ox++) {
      float x = ox * dx + x0;
      int ix = floorf(x);

      Pixel ref;
      if (ix < 0 || iy < 0 || ix >= surf_cpu.width || iy >= surf_cpu.height) {
        ref = border;
      } else {
        for (int c = 0; c < surf_cpu.channels; c++)
          ref[c] = surf_cpu(ix, iy, c);
      }
      Pixel pixel;
      for (int c = 0; c< surf_cpu.channels; c++)
        pixel[c] = out_cpu(ox, oy, c);
      EXPECT_EQ(pixel, ref) << " mismatch at (" << x << ",  " << y << ")";
    }
  }
}

TEST(SamplerGPU, Linear) {
  SamplerTestData sd;
  auto surf_cpu = sd.GetSurface(false);
  auto surf_gpu = sd.GetSurface(true);

  ASSERT_EQ(surf_cpu.channels, 3);
  ASSERT_EQ(surf_gpu.channels, 3);
  Sampler<DALI_INTERP_LINEAR, uint8_t> sampler(surf_gpu);
  Sampler<DALI_INTERP_LINEAR, uint8_t> sampler_cpu(surf_cpu);

  uint8_t border_value = 50;
  Pixel border = { border_value, border_value, border_value };

  float dy = 0.125f;
  float dx = 0.125f;
  float x0 = -1;
  float y0 = -1;

  int h = (surf_cpu.height+2) / dy + 1;
  int w = (surf_cpu.width+2) / dx + 1;
  int c = surf_cpu.channels;

  auto out_mem = memory::alloc_unique<uint8_t>(AllocType::GPU, w*h*c);
  Surface2D<uint8_t> out_surf = { out_mem.get(), w, h, c, c, w*c, 1 };

  std::vector<uint8_t> out_mem_cpu(w*h*c);

  dim3 block(32, 32);
  dim3 grid((w+31)/32, (h+31)/32);
  RunSampler<<<grid, block>>>(out_surf, sampler, border_value, dx, dy, x0, y0);
  Surface2D<uint8_t> out_cpu = { out_mem_cpu.data(), w, h, c, c, w*c, 1 };

  cudaMemcpy(out_cpu.data, out_surf.data, w*h*c, cudaMemcpyDeviceToHost);

  for (int oy = 0; oy < h; oy++) {
    float y = oy * dy + y0;
    for (int ox = 0; ox < w; ox++) {
      float x = ox * dx + x0;

      Pixel ref;
      sampler_cpu(ref.data(), x, y, border.data());

      Pixel pixel;
      for (int c = 0; c< surf_cpu.channels; c++)
        pixel[c] = out_cpu(ox, oy, c);
      EXPECT_EQ(pixel, ref) << " mismatch at (" << x << ",  " << y << ")";
    }
  }
}

}  // namespace kernels
}  // namespace dali
