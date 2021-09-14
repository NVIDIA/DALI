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
#include "dali/core/mm/memory.h"
#include "dali/kernels/imgproc/resample/resampling_filters.cuh"
#include "dali/kernels/imgproc/resample/resampling_windows.h"

namespace dali {
namespace kernels {

TEST(ResamplingFilters, GetFilters) {
  auto filters = GetResamplingFilters();
  EXPECT_NE(filters, nullptr);
  auto filters2 = GetResamplingFilters();
  EXPECT_EQ(filters, filters2);
}

__global__ void GetFilterValues(float *out, ResamplingFilter filter,
                                int n, float start, float step) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i >= n)
    return;
  out[i] = filter((i+start+filter.anchor)*filter.scale*step);
}

// DISABLED: this 'test' is only for eyeballing the filters; it hardly tests anything,
//           but would flood the output with a lot of numbers.
TEST(ResamplingFilters, DISABLED_PrintFilters) {
  auto filters = GetResamplingFilters();
  ASSERT_NE(filters, nullptr);
  for (auto &f : filters->filters) {
    for (int i = 0; i < f.num_coeffs; i++)
      std::cout << f.coeffs[i] << " ";
    std::cout << "\n\n";
  }
}


TEST(ResamplingFilters, TestTriangular) {
  auto filters = GetResamplingFilters();
  ASSERT_NE(filters, nullptr);
  int radius = 64;
  auto f = filters->Triangular(radius);
  int size = f.support();
  auto mem = mm::alloc_raw_unique<float, mm::memory_kind::device>(size);
  GetFilterValues<<<1, size>>>(mem.get(), f, size, -f.anchor, 1.0f);
  std::vector<float> host(size);
  CUDA_CALL(cudaMemcpy(host.data(), mem.get(), size*sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < size; i++) {
    float x = (i - f.anchor) * f.scale;
    float ref = std::max(0.0f, 1.0f - fabsf(x));
    EXPECT_NEAR(host[i], ref, 1e-6f);
  }
}

TEST(ResamplingFilters, Gaussian) {
  auto filters = GetResamplingFilters();
  auto host_filters = GetResamplingFiltersCPU();
  ASSERT_NE(filters, nullptr);
  int radius = 64;
  const float sigma = radius / (2 * sqrt(2));
  auto flt = filters->Gaussian(sigma);
  auto flt_host = host_filters->Gaussian(sigma);
  ASSERT_EQ(flt.anchor, flt_host.anchor);
  ASSERT_EQ(flt.scale, flt_host.scale);
  ASSERT_EQ(flt.num_coeffs, flt_host.num_coeffs);
  int size = flt.support();
  auto mem = mm::alloc_raw_unique<float, mm::memory_kind::device>(size);
  GetFilterValues<<<1, size>>>(mem.get(), flt, size, -radius, 1);
  std::vector<float> host(size);
  CUDA_CALL(cudaMemcpy(host.data(), mem.get(), size*sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < size-1; i++) {
    float x = (i - radius);
    float ref = expf(-x*x / (2 * sigma*sigma));
    float cpu = flt_host((x + flt.anchor) * flt.scale);
    EXPECT_NEAR(host[i], ref, 1e-3f) << "@ i = " << i << " x = " << x;;
    EXPECT_NEAR(host[i], cpu, 1e-6f) << "@ i = " << i << " x = " << x;;
  }
}

TEST(ResamplingFilters, Lanczos3) {
  auto filters = GetResamplingFilters();
  auto host_filters = GetResamplingFiltersCPU();
  ASSERT_NE(filters, nullptr);
  int radius = 3*64;
  auto flt = filters->Lanczos3();
  auto flt_host = host_filters->Lanczos3();
  ASSERT_EQ(flt.anchor, flt_host.anchor);
  ASSERT_EQ(flt.scale, flt_host.scale);
  ASSERT_EQ(flt.num_coeffs, flt_host.num_coeffs);
  int size = 2*radius+1;
  auto mem = mm::alloc_raw_unique<float, mm::memory_kind::device>(size);
  GetFilterValues<<<1, size>>>(mem.get(), flt, size, -3.0f, 3.0f / radius);
  std::vector<float> host(size);
  CUDA_CALL(cudaMemcpy(host.data(), mem.get(), size*sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < size; i++) {
    float x = 3.0f * (i - radius) / radius;
    float ref = sinc(x) * sinc(x / 3);
    float cpu = flt_host((x + flt.anchor) * flt.scale);
    EXPECT_NEAR(host[i], ref, 1e-3f) << "@ i = " << i << " x = " << x;
    EXPECT_NEAR(host[i], cpu, 1e-6f) << "@ i = " << i << " x = " << x;;
  }
}

TEST(ResamplingFilters, Cubic) {
  auto filters = GetResamplingFilters();
  auto host_filters = GetResamplingFiltersCPU();
  ASSERT_NE(filters, nullptr);
  int radius = 64;
  auto flt = filters->Cubic();
  auto flt_host = host_filters->Cubic();
  ASSERT_EQ(flt.anchor, flt_host.anchor);
  ASSERT_EQ(flt.scale, flt_host.scale);
  ASSERT_EQ(flt.num_coeffs, flt_host.num_coeffs);
  int size = 2*radius+1;
  auto mem = mm::alloc_raw_unique<float, mm::memory_kind::device>(size);
  GetFilterValues<<<1, size>>>(mem.get(), flt, size, -2.0f, 2.0f / radius);
  std::vector<float> host(size);
  CUDA_CALL(cudaMemcpy(host.data(), mem.get(), size*sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < size; i++) {
    float x = 2.0f * (i - radius) / radius;
    float ref = CubicWindow(x);
    float cpu = flt_host((x + flt.anchor) * flt.scale);
    EXPECT_NEAR(host[i], ref, 1e-3f) << "@ i = " << i << " x = " << x;
    EXPECT_NEAR(host[i], cpu, 1e-6f) << "@ i = " << i << " x = " << x;;
  }
}

}  // namespace kernels
}  // namespace dali
