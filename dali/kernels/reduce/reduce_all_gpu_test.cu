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

#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "dali/kernels/reduce/reduce_all_gpu_impl.cuh"
#include "dali/core/util.h"
#include "dali/kernels/alloc.h"
#include "dali/core/span.h"
#include "dali/core/cuda_event.h"

namespace dali {
namespace kernels {

template <typename T>
double RefSum(span<T> in) {
  switch (in.size()) {
    case 0:
      return 0;
    case 1:
      return in[0];
    default: {
      int m = in.size() / 2;
      int n = in.size() - m;
      return RefSum(make_span(in.data(), m)) + RefSum(make_span(in.data() + m, n));
    }
  }
}

TEST(ReduceGPU, ReduceAllKernel) {
  std::mt19937_64 rng(1234);
  std::uniform_real_distribution<float> dist(0, 1);

  int n_in = 1<<24;  // 16M numbers
  dim3 block(32, 32);
  int nblock = 1024;
  int n_out = std::min<int>(div_ceil(n_in, nblock), 1024);
  auto in_data = memory::alloc_unique<float>(AllocType::GPU, n_in);
  auto out_data = memory::alloc_unique<float>(AllocType::GPU, n_out);
  std::vector<float> in_cpu(n_in), out_cpu(n_out);
  for (auto &x : in_cpu)
    x = dist(rng);
  double ref_sum = RefSum(make_cspan(in_cpu));

  cudaMemcpy(in_data.get(), in_cpu.data(), n_in * sizeof(*in_data), cudaMemcpyHostToDevice);

  dim3 grid = n_out;
  ReduceAllKernel<<<1, block>>>(out_data.get(), in_data.get(), n_in);
  cudaDeviceSynchronize();
  auto start = CUDAEvent::CreateWithFlags(0);
  auto end =   CUDAEvent::CreateWithFlags(0);
  cudaEventRecord(start);
  ReduceAllKernel<<<grid, block>>>(out_data.get(), in_data.get(), n_in);
  cudaEventRecord(end);
  cudaMemcpy(out_cpu.data(), out_data.get(), n_out * sizeof(*out_data), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  float t = 0;
  cudaEventElapsedTime(&t, start, end);
  double out_sum = RefSum(make_cspan(out_cpu));
  EXPECT_NEAR(out_sum, ref_sum, ref_sum * 1e-7 + 1e-7);

  t /= 1000;
  std::cout << n_in * sizeof(*in_data) / t * 1e-9 << " GB/s" << std::endl;
}

TEST(ReduceGPU, ReduceAllBatchedKernel) {
  std::mt19937_64 rng(1234);
  std::uniform_real_distribution<float> dist(0, 1);
  std::uniform_int_distribution<int> size_dist(10000, 1000000);

  int64_t n_in = 0;
  int samples = 35;
  std::vector<int64_t> sizes(samples);
  for (auto &size : sizes) {
    size = size_dist(rng);
    n_in += size;
  }

  dim3 block(32, 32);
  int nout_per_sample = 32;
  int n_out = samples * nout_per_sample;
  dim3 grid(nout_per_sample, samples);
  auto in_data = memory::alloc_unique<float>(AllocType::GPU, n_in);
  auto out_data = memory::alloc_unique<float>(AllocType::GPU, n_out);
  std::vector<float> in_cpu(n_in), out_cpu(n_out);
  for (auto &x : in_cpu)
    x = dist(rng);

  auto gpu_dev_ptrs = memory::alloc_unique<const float*>(AllocType::GPU, samples);
  auto gpu_sizes = memory::alloc_unique<int64_t>(AllocType::GPU, samples);
  vector<const float *> host_ptrs(samples);
  vector<const float *> cpu_dev_ptrs(samples);
  int64_t offset = 0;
  for (int i = 0; i < samples; i++) {
    host_ptrs[i] = in_cpu.data() + offset;
    cpu_dev_ptrs[i] = in_data.get() + offset;
    offset += sizes[i];
  }

  // data
  cudaMemcpy(in_data.get(), in_cpu.data(), n_in * sizeof(*in_data), cudaMemcpyHostToDevice);
  // pointers to sample data
  cudaMemcpy(gpu_dev_ptrs.get(), cpu_dev_ptrs.data(), samples * sizeof(*gpu_dev_ptrs),
             cudaMemcpyHostToDevice);
  // sample sizes
  cudaMemcpy(gpu_sizes.get(), sizes.data(), samples * sizeof(*gpu_sizes), cudaMemcpyHostToDevice);

  // warm-up
  ReduceAllBatchedKernel<<<1, block>>>(out_data.get(), gpu_dev_ptrs.get(), gpu_sizes.get());
  cudaDeviceSynchronize();
  auto start = CUDAEvent::CreateWithFlags(0);
  auto end =   CUDAEvent::CreateWithFlags(0);
  cudaEventRecord(start);
  ReduceAllBatchedKernel<<<grid, block>>>(out_data.get(), gpu_dev_ptrs.get(), gpu_sizes.get());
  cudaEventRecord(end);
  cudaMemcpy(out_cpu.data(), out_data.get(), n_out * sizeof(*out_data), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  float t = 0;
  cudaEventElapsedTime(&t, start, end);

  offset = 0;
  for (int i = 0; i < samples; i++) {
    double ref_sum = RefSum(make_span(host_ptrs[i], sizes[i]));
    double out_sum = RefSum(make_span(&out_cpu[i*nout_per_sample], nout_per_sample));
    EXPECT_NEAR(ref_sum, out_sum, ref_sum * 1e-7 + 1e-7);
    offset += sizes[i];
  }

  t /= 1000;
  std::cout << n_in * sizeof(*in_data) / t * 1e-9 << " GB/s" << std::endl;
}

}  // namespace kernels
}  // namespace dali
