// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
using PixelF = std::array<float, 3>;

template <typename Out, DALIInterpType interp, int MaxChannels = 8, typename In>
__global__ void RunSampler2D(
      Surface2D<Out> out, Sampler2D<interp, In> sampler, In border_value,
      float dx, float dy, float x0, float y0) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= out.size.x || y >= out.size.y)
    return;

  In border[3] = { border_value, border_value, border_value };

  vec2 src = { x*dx + x0, y*dy + y0 };
  Out tmp[MaxChannels];
  sampler(tmp, src, border);
  for (int c = 0; c < out.channels; c++)
    out(x, y, c) = tmp[c];
}

template <typename Out, DALIInterpType interp, int MaxChannels = 8, typename In>
__global__ void RunSampler3D(
      Surface3D<Out> out, Sampler3D<interp, In> sampler, In border_value,
      float dx, float dy, float dz, float x0, float y0, float z0) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  if (x >= out.size.x || y >= out.size.y || z >= out.size.z)
    return;

  In border[3] = { border_value, border_value, border_value };

  vec3 src = { x*dx + x0, y*dy + y0, z*dz + z0 };
  Out tmp[MaxChannels];
  sampler(tmp, src, border);
  for (int c = 0; c < out.channels; c++)
    out(x, y, z, c) = tmp[c];
}

TEST(Sampler2D_GPU, NN) {
  SamplerTestData<uint8_t> sd;
  auto surf_cpu = sd.GetSurface2D(false);
  auto surf_gpu = sd.GetSurface2D(true);

  ASSERT_EQ(surf_cpu.channels, 3);
  ASSERT_EQ(surf_gpu.channels, 3);
  Sampler2D<DALI_INTERP_NN, uint8_t> sampler(surf_gpu);

  uint8_t border_value = 50;
  Pixel border = { border_value, border_value, border_value };

  float dy = 0.125f;
  float dx = 0.125f;
  float x0 = -1;
  float y0 = -1;

  int w = (surf_cpu.size.x+2) / dx + 1;
  int h = (surf_cpu.size.y+2) / dy + 1;
  int c = surf_cpu.channels;

  auto out_mem = mm::alloc_raw_unique<uint8_t, mm::memory_kind::device>(w*h*c);
  Surface2D<uint8_t> out_surf = { out_mem.get(), { w, h }, c, { c, w*c }, 1 };

  std::vector<uint8_t> out_mem_cpu(w*h*c);

  dim3 block(32, 32);
  dim3 grid((w+31)/32, (h+31)/32);
  RunSampler2D<<<grid, block>>>(out_surf, sampler, border_value, dx, dy, x0, y0);
  Surface2D<uint8_t> out_cpu = { out_mem_cpu.data(), { w, h }, c, { c, w*c }, 1 };

  CUDA_CALL(cudaMemcpy(out_cpu.data, out_surf.data, w*h*c, cudaMemcpyDeviceToHost));

  for (int oy = 0; oy < h; oy++) {
    float y = oy * dy + y0;
    int iy = floorf(y);
    for (int ox = 0; ox < w; ox++) {
      float x = ox * dx + x0;
      int ix = floorf(x);

      Pixel ref;
      if (ix < 0 || iy < 0 || ix >= surf_cpu.size.x || iy >= surf_cpu.size.y) {
        ref = border;
      } else {
        for (int c = 0; c < surf_cpu.channels; c++)
          ref[c] = surf_cpu(ix, iy, c);
      }
      Pixel pixel;
      for (int c = 0; c< surf_cpu.channels; c++)
        pixel[c] = out_cpu(ox, oy, c);
      EXPECT_EQ(pixel, ref) << " mismatch at " << vec2(x, y);
    }
  }
}


TEST(Sampler3D_GPU, NN) {
  SamplerTestData<uint8_t> sd;
  auto surf_cpu = sd.GetSurface3D(false);
  auto surf_gpu = sd.GetSurface3D(true);

  ASSERT_EQ(surf_cpu.channels, 3);
  ASSERT_EQ(surf_gpu.channels, 3);
  Sampler3D<DALI_INTERP_NN, uint8_t> sampler(surf_gpu);

  uint8_t border_value = 50;
  Pixel border = { border_value, border_value, border_value };

  float dy = 0.125f;
  float dx = 0.125f;
  float dz = 0.25f;
  float x0 = -1;
  float y0 = -1;
  float z0 = -1;

  int w = (surf_cpu.size.x+2) / dx + 1;
  int h = (surf_cpu.size.y+2) / dy + 1;
  int d = (surf_cpu.size.z+2) / dz + 1;
  int c = surf_cpu.channels;

  auto out_mem = mm::alloc_raw_unique<uint8_t, mm::memory_kind::device>(w*h*d*c);
  Surface3D<uint8_t> out_surf = { out_mem.get(), { w, h, d }, c, { c, w*c, h*w*c }, 1 };

  std::vector<uint8_t> out_mem_cpu(w*h*d*c);

  dim3 block(32, 32, 1);
  dim3 grid((w+31)/32, (h+31)/32, d);
  RunSampler3D<<<grid, block>>>(out_surf, sampler, border_value, dx, dy, dz, x0, y0, z0);
  Surface3D<uint8_t> out_cpu = { out_mem_cpu.data(), { w, h, d }, c, { c, w*c, h*w*c }, 1 };

  CUDA_CALL(cudaMemcpy(out_cpu.data, out_surf.data, w*h*d*c, cudaMemcpyDeviceToHost));

  for (int oz = 0; oz < d; oz++) {
    float z = oz * dz + z0;
    int iz = floorf(z);
    for (int oy = 0; oy < h; oy++) {
      float y = oy * dy + y0;
      int iy = floorf(y);
      for (int ox = 0; ox < w; ox++) {
        float x = ox * dx + x0;
        int ix = floorf(x);

        Pixel ref;
        if (ix < 0 || iy < 0 || iz < 0 ||
            ix >= surf_cpu.size.x || iy >= surf_cpu.size.y || iz >= surf_cpu.size.z) {
          ref = border;
        } else {
          for (int c = 0; c < surf_cpu.channels; c++)
            ref[c] = surf_cpu(ix, iy, iz, c);
        }
        Pixel pixel;
        for (int c = 0; c< surf_cpu.channels; c++)
          pixel[c] = out_cpu(ox, oy, oz, c);
        EXPECT_EQ(pixel, ref) << " mismatch at " << vec3(x, y, z);
      }
    }
  }
}



TEST(Sampler2D_GPU, Linear) {
  SamplerTestData<uint8_t> sd;
  auto surf_cpu = sd.GetSurface2D(false);
  auto surf_gpu = sd.GetSurface2D(true);

  ASSERT_EQ(surf_cpu.channels, 3);
  ASSERT_EQ(surf_gpu.channels, 3);
  Sampler2D<DALI_INTERP_LINEAR, uint8_t> sampler(surf_gpu);
  Sampler2D<DALI_INTERP_LINEAR, uint8_t> sampler_cpu(surf_cpu);

  uint8_t border_value = 50;
  Pixel border = { border_value, border_value, border_value };

  float dy = 0.125f;
  float dx = 0.125f;
  float x0 = -1;
  float y0 = -1;

  int w = (surf_cpu.size.x+2) / dx + 1;
  int h = (surf_cpu.size.y+2) / dy + 1;
  int c = surf_cpu.channels;

  auto out_mem = mm::alloc_raw_unique<uint8_t, mm::memory_kind::device>(w*h*c);
  Surface2D<uint8_t> out_surf = { out_mem.get(), { w, h }, c, { c, w*c}, 1 };

  std::vector<uint8_t> out_mem_cpu(w*h*c);

  dim3 block(32, 32);
  dim3 grid((w+31)/32, (h+31)/32);
  RunSampler2D<<<grid, block>>>(out_surf, sampler, border_value, dx, dy, x0, y0);
  Surface2D<uint8_t> out_cpu = { out_mem_cpu.data(), { w, h }, c, { c, w*c }, 1 };

  CUDA_CALL(cudaMemcpy(out_cpu.data, out_surf.data, w*h*c, cudaMemcpyDeviceToHost));

  const float eps = 0.50000025f;  // 0.5 + 4 ULP

  for (int oy = 0; oy < h; oy++) {
    float y = oy * dy + y0;
    for (int ox = 0; ox < w; ox++) {
      float x = ox * dx + x0;

      vec2 pos = { x, y };

      PixelF ref;
      sampler_cpu(ref.data(), pos, border.data());

      Pixel pixel;
      for (int c = 0; c< surf_cpu.channels; c++) {
        pixel[c] = out_cpu(ox, oy, c);
        EXPECT_NEAR(pixel[c], ref[c], eps) << " mismatch at " << pos;
      }
    }
  }
}

TEST(Sampler3D_GPU, Linear) {
  SamplerTestData<uint8_t> sd;
  auto surf_cpu = sd.GetSurface3D(false);
  auto surf_gpu = sd.GetSurface3D(true);

  ASSERT_EQ(surf_cpu.channels, 3);
  ASSERT_EQ(surf_gpu.channels, 3);
  Sampler3D<DALI_INTERP_LINEAR, uint8_t> sampler(surf_gpu);
  Sampler3D<DALI_INTERP_LINEAR, uint8_t> sampler_cpu(surf_cpu);

  uint8_t border_value = 50;
  Pixel border = { border_value, border_value, border_value };

  float dy = 0.125f;
  float dx = 0.125f;
  float dz = 0.25f;
  float x0 = -1;
  float y0 = -1;
  float z0 = -1;

  int w = (surf_cpu.size.x+2) / dx + 1;
  int h = (surf_cpu.size.y+2) / dy + 1;
  int d = (surf_cpu.size.z+2) / dz + 1;
  int c = surf_cpu.channels;

  auto out_mem = mm::alloc_raw_unique<uint8_t, mm::memory_kind::device>(w*h*d*c);
  Surface3D<uint8_t> out_surf = { out_mem.get(), { w, h, d }, c, { c, w*c, h*w*c }, 1 };

  std::vector<uint8_t> out_mem_cpu(w*h*d*c);

  dim3 block(32, 32, 1);
  dim3 grid((w+31)/32, (h+31)/32, d);
  RunSampler3D<<<grid, block>>>(out_surf, sampler, border_value, dx, dy, dz, x0, y0, z0);
  Surface3D<uint8_t> out_cpu = { out_mem_cpu.data(), { w, h, d }, c, { c, w*c, h*w*c }, 1 };

  CUDA_CALL(cudaMemcpy(out_cpu.data, out_surf.data, w*h*d*c, cudaMemcpyDeviceToHost));

  const float eps = 0.50000025f;  // 0.5 + 4 ULP

  for (int oz = 0; oz < d; oz++) {
    float z = oz * dz + z0;
    for (int oy = 0; oy < h; oy++) {
      float y = oy * dy + y0;
      for (int ox = 0; ox < w; ox++) {
        float x = ox * dx + x0;

        vec3 pos = { x, y, z };

        PixelF ref;
        sampler_cpu(ref.data(), pos, border.data());

        Pixel pixel;
        for (int c = 0; c< surf_cpu.channels; c++) {
          pixel[c] = out_cpu(ox, oy, oz, c);
          EXPECT_NEAR(pixel[c], ref[c], eps) << " mismatch at " << pos;
        }
      }
    }
  }
}

}  // namespace kernels
}  // namespace dali
