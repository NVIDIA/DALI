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
#include "dali/kernels/alloc.h"
#include "dali/kernels/imgproc/resample/resampling_filters.cuh"

namespace dali {
namespace kernels {

TEST(ResamplingFilters, GetFilters) {
  auto filters = GetResamplingFilters(0);
  EXPECT_NE(filters, nullptr);
  auto filters2 = GetResamplingFilters(0);
  EXPECT_EQ(filters, filters2);
}

__global__ void GetFilterValues(float *out, ResamplingFilter filter,
                                int n, float start, float step) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i >= n)
    return;
  out[i] = filter((i+start+filter.anchor)*filter.scale*step);
}

TEST(ResamplingFilters, DISABLED_PrintFilters) {
  auto filters = GetResamplingFilters(0);
  ASSERT_NE(filters, nullptr);
  for (auto &f : filters->filters) {
    int size = f.num_coeffs;
    auto mem = memory::alloc_unique<float>(AllocType::GPU, size);
    GetFilterValues<<<1, size>>>(mem.get(), f, size, -f.anchor, 1);
    std::vector<float> host(size);
    cudaMemcpy(host.data(), mem.get(), size*sizeof(float), cudaMemcpyDeviceToHost);
    for (auto &value : host)
      std::cout << value << " ";
    std::cout << "\n\n";
  }
}


TEST(ResamplingFilters, TestTriangular) {
  auto filters = GetResamplingFilters(0);
  ASSERT_NE(filters, nullptr);
  int radius = 64;
  auto f = filters->Triangular(radius);
  int size = f.support();
  auto mem = memory::alloc_unique<float>(AllocType::GPU, size);
  GetFilterValues<<<1, size>>>(mem.get(), f, size, -f.anchor, 1.0f);
  std::vector<float> host(size);
  cudaMemcpy(host.data(), mem.get(), size*sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; i++) {
    float x = (i - f.anchor) * f.scale;
    float ref = std::max(0.0f, 1.0f - fabsf(x));
    EXPECT_NEAR(host[i], ref, 1e-6f);
  }
}

TEST(ResamplingFilters, Gaussian) {
  auto filters = GetResamplingFilters(0);
  ASSERT_NE(filters, nullptr);
  int radius = 64;
  const float sigma = radius / (2 * sqrt(2));
  auto f = filters->Gaussian(sigma);
  int size = f.support();
  auto mem = memory::alloc_unique<float>(AllocType::GPU, size);
  GetFilterValues<<<1, size>>>(mem.get(), f, size, -radius, 1);
  std::vector<float> host(size);
  cudaMemcpy(host.data(), mem.get(), size*sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < size-1; i++) {
    float x = (i - radius);
    float ref = expf(-x*x / (2 * sigma*sigma));
    float cpu = f((x + f.anchor) * f.scale);
    EXPECT_NEAR(host[i], ref, 1e-3f);
    EXPECT_NEAR(host[i], cpu, 1e-6f);
  }
}


}  // namespace kernels
}  // namespace dali
