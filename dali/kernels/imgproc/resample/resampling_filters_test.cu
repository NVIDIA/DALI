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

__global__ void GetFilterValues(float *out, ResamplingFilter filter, int n, float step) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i >= n)
    return;
  out[i] = filter.at_abs(i*step);
}

TEST(ResamplingFilters, PrintFilters) {
  auto filters = GetResamplingFilters(0);
  ASSERT_NE(filters, nullptr);
  for (auto &f : filters->filters) {
    int size = f.num_coeffs;
    auto mem = memory::alloc_unique<float>(AllocType::GPU, size);
    GetFilterValues<<<1, size>>>(mem.get(), f, size, 1);
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
  int scale = 64;
  int size = 2 * scale + 1;
  auto mem = memory::alloc_unique<float>(AllocType::GPU, size);
  auto &f = filters->filters[0];
  GetFilterValues<<<1, size>>>(mem.get(), f, size, 1.0f/scale);
  std::vector<float> host(size);
  cudaMemcpy(host.data(), mem.get(), size*sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; i++) {
    float ref = 1.0f - fabsf(i - scale) / scale;
    EXPECT_NEAR(host[i], ref, 1e-6f);
  }
}


}  // namespace kernels
}  // namespace dali
