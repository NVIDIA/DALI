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

#ifndef DALI_KERNELS_IMGPROC_SAMPLER_TEST_H_
#define DALI_KERNELS_IMGPROC_SAMPLER_TEST_H_

#include <gtest/gtest.h>
#include <random>
#include "dali/core/mm/memory.h"
#include "dali/kernels/imgproc/sampler.h"
#include "dali/kernels/imgproc/surface.h"

namespace dali {
namespace kernels {

template <typename T>
struct SamplerTestData {
  static constexpr int D = 3;
  static constexpr int W = 4;
  static constexpr int H = 5;
  static constexpr int C = 3;

  Surface2D<const T> GetSurface2D(bool gpu) {
    Surface2D<const T> surface;
    surface.data = gpu ? GetGPUData() : GetCPUData();
    surface.size = { W, H };
    surface.channels = C;
    surface.strides = { C, W*C };
    surface.channel_stride = 1;
    return surface;
  }

  Surface3D<const T> GetSurface3D(bool gpu) {
    Surface3D<const T> surface;
    surface.data = gpu ? GetGPUData() : GetCPUData();
    surface.size = { W, H, D };
    surface.channels = C;
    surface.strides = { C, W*C, H*W*C };
    surface.channel_stride = 1;
    return surface;
  }

 private:
  static const T *InitCPUData() {
    static T data[D*W*H*C];
    std::mt19937_64 rng;
    if (std::is_integral<T>::value) {
      std::uniform_int_distribution<int> dist(0, 0x7fffffff);
      for (T &x : data)
        x = T(dist(rng));
    } else {
      std::uniform_real_distribution<double> dist(-1, 1);
      for (T &x : data)
        x = T(dist(rng));
    }
    return data;
  }

  static const T *GetCPUData() {
    static const T *data = InitCPUData();  // use magic static to run the InitCPUData() once
    return data;
  }

  const T *GetGPUData() {
    if (!gpu_storage) {
      gpu_storage = mm::alloc_raw_unique<T, mm::memory_kind::device>(D*W*H*C);
      CUDA_CALL(
        cudaMemcpy(gpu_storage.get(), GetCPUData(), sizeof(T)*D*W*H*C, cudaMemcpyHostToDevice));
    }
    return gpu_storage.get();
  }

  mm::uptr<T> gpu_storage;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_SAMPLER_TEST_H_
