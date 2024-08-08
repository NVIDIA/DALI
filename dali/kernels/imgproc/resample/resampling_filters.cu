// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cassert>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <utility>
#include "dali/kernels/imgproc/resample/resampling_filters.cuh"
#include "dali/kernels/imgproc/resample/resampling_windows.h"
#include "dali/core/mm/memory.h"
#include "dali/core/span.h"
#include "dali/core/cuda_stream_pool.h"

namespace dali {
namespace kernels {

template <typename Function>
inline void InitFilter(ResamplingFilter &filter, Function F) {
  for (int i = 0; i < filter.num_coeffs; i++)
    filter.coeffs[i] = F(i);
}

void InitGaussianFilter(ResamplingFilter filter) {
  InitFilter(filter, [&](int i) {
    float x = 4 * (i - (filter.num_coeffs-1)*0.5f) / (filter.num_coeffs-1);
    return expf(-x*x);
  });
}

void InitLanczosFilter(ResamplingFilter filter, float a) {
  InitFilter(filter, [&](int i) {
    float x = 2 * a * (i - (filter.num_coeffs-1)*0.5f) / (filter.num_coeffs-1);
    return LanczosWindow(x, a);
  });
}

void InitCubicFilter(ResamplingFilter filter) {
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

template <typename MemoryKind>
void InitFilters(ResamplingFilters &filters) {
  const int lanczos_resolution = 32;
  const int lanczos_a = 3;
  const int triangular_size = 3;
  const int gaussian_size = 65;
  const int cubic_size = 129;
  const int lanczos_size = (2*lanczos_a*lanczos_resolution + 1);
  const int total_size = triangular_size + gaussian_size + cubic_size + lanczos_size;

  constexpr bool need_staging = !mm::is_host_accessible<MemoryKind>;

  using tmp_kind = std::conditional_t<need_staging, mm::memory_kind::host, MemoryKind>;
  filters.filter_data = mm::alloc_raw_unique<float, tmp_kind>(total_size);

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
  assert(filters.filters.back().coeffs + filters.filters.back().num_coeffs -
         filters.filter_data.get() <= total_size);

  auto *tri_coeffs = filters.filters[Idx_Triangular].coeffs;
  tri_coeffs[0] = 0;
  tri_coeffs[1] = 1;
  tri_coeffs[2] = 0;

  InitGaussianFilter(filters.filters[Idx_Gaussian]);
  InitLanczosFilter(filters.filters[Idx_Lanczos3], lanczos_a);
  InitCubicFilter(filters.filters[Idx_Cubic]);

  filters[2].rescale(6);
  filters[3].rescale(4);

  if (need_staging) {
    auto cuda_stream = CUDAStreamPool::instance().Get();
    auto filter_data_gpu = mm::alloc_raw_async_unique<float, mm::memory_kind::device>(
        total_size, cuda_stream, mm::host_sync);
    CUDA_CALL(cudaMemcpyAsync(filter_data_gpu.get(), filters.filter_data.get(),
                              total_size * sizeof(float), cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CALL(cudaStreamSynchronize(cuda_stream));
    ptrdiff_t diff = filter_data_gpu.get() - filters.filter_data.get();
    filters.filter_data = std::move(filter_data_gpu);
    for (auto &f : filters.filters)
      f.coeffs += diff;
  }
}

ResamplingFilter ResamplingFilters::Cubic(float radius) const noexcept {
  auto flt = filters[Idx_Cubic];
  flt.rescale(2.0f * std::max(2.0f, radius));
  return flt;
}

ResamplingFilter ResamplingFilters::Gaussian(float sigma) const noexcept {
  auto flt = filters[Idx_Gaussian];
  flt.rescale(std::max(1.0f, static_cast<float>(4*M_SQRT2)*sigma));
  return flt;
}

ResamplingFilter ResamplingFilters::Lanczos3(float radius) const noexcept {
  auto flt = filters[Idx_Lanczos3];
  flt.rescale(2.0f * std::max(3.0f, radius));
  return flt;
}

ResamplingFilter ResamplingFilters::Triangular(float radius) const noexcept {
  auto flt = filters[Idx_Triangular];
  flt.rescale(std::max(1.0f, 2*radius));
  return flt;
}



std::shared_ptr<ResamplingFilters> GetResamplingFilters() {
  (void)mm::GetDefaultDeviceResource();
  static std::mutex filter_mutex;
  static std::vector<std::weak_ptr<ResamplingFilters>> filters;
  std::lock_guard<std::mutex> lock(filter_mutex);
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess)
    return nullptr;

  if (filters.empty()) {
    int count;
    cudaGetDeviceCount(&count);
    filters.resize(count);
  }

  auto ptr = filters[device].lock();
  if (!ptr) {
    ptr = std::make_shared<ResamplingFilters>();
    InitFilters<mm::memory_kind::device>(*ptr);
    filters[device] = ptr;
  }
  return ptr;
}


std::shared_ptr<ResamplingFilters> GetResamplingFiltersCPU() {
  (void)mm::GetDefaultResource<mm::memory_kind::host>();
  static std::shared_ptr<ResamplingFilters> cpu_filters = []() {
    auto filters = std::make_shared<ResamplingFilters>();
    InitFilters<mm::memory_kind::host>(*filters);
    return filters;
  }();
  return cpu_filters;
}

}  // namespace kernels
}  // namespace dali
