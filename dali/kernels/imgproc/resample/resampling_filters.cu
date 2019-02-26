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

#include <cuda_runtime.h>
#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "dali/kernels/imgproc/resample/resampling_filters.cuh"
#include "dali/kernels/imgproc/resample/resampling_windows.h"
#include "dali/kernels/alloc.h"
#include "dali/kernels/span.h"

namespace dali {
namespace kernels {

template <typename Function>
__device__ inline void InitFilter(ResamplingFilter &filter, Function F) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i > filter.num_coeffs)
    return;
  filter.coeffs[i] = F(i);
}

__global__ void InitGaussianFilter(ResamplingFilter filter) {
  InitFilter(filter, [&](int i) {
    float x = 4 * (i - (filter.num_coeffs-1)*0.5f) / (filter.num_coeffs-1);
    return expf(-x*x);
  });
}

__global__ void InitLanczosFilter(ResamplingFilter filter, float a) {
  InitFilter(filter, [&](int i) {
    float x = 2 * a * (i - (filter.num_coeffs-1)*0.5f) / (filter.num_coeffs-1);
    return LanczosWindow(x, a);
  });
}

__global__ void InitCubicFilter(ResamplingFilter filter) {
  InitFilter(filter, [&](int i) {
    float x = 4 * (i - (filter.num_coeffs-1)*0.5f) / (filter.num_coeffs-1);
    return CubicWindow(x);
  });
}

enum FilterIdx {
  Idx_Triangular = 0,
  Idx_Gaussian,
  Idx_Lanczos3,
  Idx_Cubic
};

void InitFilters(ResamplingFilters &filters, cudaStream_t stream) {
  const int lanczos_resolution = 32;
  const int lanczos_a = 3;
  const int triangular_size = 3;
  const int gaussian_size = 65;
  const int cubic_size = 129;
  const int lanczos_size = (2*lanczos_a*lanczos_resolution + 1);
  const int total_size = triangular_size + gaussian_size + lanczos_size;

  filters.filter_data = memory::alloc_unique<float>(AllocType::Unified, total_size);

  auto add_filter = [&](int size) {
    float *base = filters.filters.empty()
        ? filters.filter_data.get()
        : filters.filters.back().coeffs + filters.filters.back().num_coeffs;
    filters.filters.push_back({ base, size, 1, (size - 1) * 0.5f});
  };
  add_filter(triangular_size);
  add_filter(gaussian_size);
  add_filter(lanczos_size);
  add_filter(cubic_size);

  auto *tri_coeffs = filters.filters[Idx_Triangular].coeffs;
  tri_coeffs[0] = 0;
  tri_coeffs[1] = 1;
  tri_coeffs[0] = 0;

  InitGaussianFilter<<<1, gaussian_size, 0, stream>>>(filters.filters[Idx_Gaussian]);
  InitLanczosFilter<<<1, lanczos_size, 0, stream>>>(filters.filters[Idx_Lanczos3], lanczos_a);
  InitCubicFilter<<<1, cubic_size, 0, stream>>>(filters.filters[Idx_Cubic]);

  filters[2].rescale(6);
  filters[3].rescale(4);
  cudaStreamSynchronize(stream);
}

ResamplingFilter ResamplingFilters::Cubic() const noexcept {
  return filters[Idx_Cubic];
}

ResamplingFilter ResamplingFilters::Gaussian(float sigma) const noexcept {
  auto flt = filters[Idx_Gaussian];
  flt.rescale(std::max(1.0f, static_cast<float>(4*M_SQRT2)*sigma));
  return flt;
}

ResamplingFilter ResamplingFilters::Lanczos3() const noexcept {
  return filters[Idx_Lanczos3];
}

ResamplingFilter ResamplingFilters::Triangular(float radius) const noexcept {
  auto flt = filters[Idx_Triangular];
  flt.rescale(std::max(1.0f, 2*radius));
  return flt;
}


static std::unordered_map<int, std::weak_ptr<ResamplingFilters>> filters;
static std::mutex filter_mutex;

std::shared_ptr<ResamplingFilters> GetResamplingFilters(cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(filter_mutex);
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess)
    return nullptr;

  auto ptr = filters[device].lock();
  if (!ptr) {
    ptr = std::make_shared<ResamplingFilters>();
    InitFilters(*ptr, stream);
    filters[device] = ptr;
  }
  return ptr;
}

}  // namespace kernels
}  // namespace dali
