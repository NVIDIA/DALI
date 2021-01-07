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

#ifndef DALI_OPERATORS_RANDOM_RNG_BASE_GPU_CUH_
#define DALI_OPERATORS_RANDOM_RNG_BASE_GPU_CUH_

#include <utility>
#include <vector>
#include "dali/core/convert.h"
#include "dali/kernels/alloc.h"
#include "dali/operators/random/rng_base_gpu.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/batch_rng.h"
#include "dali/core/static_switch.h"

namespace dali {

namespace {

template <typename Out, typename Dist, bool DefaultDist = false>
__global__ void RNGKernel(BlockDesc* __restrict__ block_descs,
                          curandState* __restrict__ states,
                          const Dist* __restrict__ dists, int nblocks) {
  int block_size = blockDim.x * blockDim.y;
  int local_tid = blockDim.x * threadIdx.y + threadIdx.x;
  int64_t global_tid = blockIdx.y * block_size + local_tid;
  auto rng = states + global_tid;
  int blk_stride = blockDim.y * gridDim.y;
  int blk = blockIdx.y * blockDim.y + threadIdx.y;
  for (; blk < nblocks; blk += blk_stride) {
    auto desc = block_descs[blk];
    auto start = static_cast<Out*>(desc.start);
    auto block_end = start + desc.size;
    Dist dist = DefaultDist ? Dist() : dists[desc.sample_idx];
    for (auto out = start + threadIdx.x; out < block_end; out += blockDim.x) {
      *out = ConvertSat<Out>(dist(rng));
    }
  }
}

}  // namespace

template <typename Backend, typename Impl>
template <typename T, typename Dist>
void RNGBase<Backend, Impl>::RunImplTyped(workspace_t<GPUBackend> &ws) {
  static_assert(std::is_same<Backend, GPUBackend>::value, "Unexpected backend");
  auto out_view = view<T>(ws.template OutputRef<GPUBackend>(0));
  int nsamples = out_view.shape.size();
  auto blocks_cpu = backend_data_.block_descs_cpu_.get();
  auto blocks_gpu = backend_data_.block_descs_gpu_.get();
  auto rngs = backend_data_.randomizer_.states();
  int block_sz = backend_data_.block_size_;
  int max_nblocks = backend_data_.max_blocks_;
  int blockdesc_count = -1;
  blockdesc_count = SetupBlockDescs(
    blocks_cpu, block_sz, max_nblocks, out_view, ws.stream());
  if (blockdesc_count == 0) {
    return;
  }

  int64_t blockdesc_max_sz = -1;
  for (int b = 0; b < blockdesc_count; b++) {
    int64_t block_sz = blocks_cpu[b].size;
    if (block_sz > blockdesc_max_sz)
      blockdesc_max_sz = block_sz;
  }

  cudaMemcpyAsync(blocks_gpu, blocks_cpu,
                  sizeof(BlockDesc) * blockdesc_count, cudaMemcpyHostToDevice, ws.stream());

  auto &dists_cpu = backend_data_.dists_cpu_;
  auto &dists_gpu = backend_data_.dists_gpu_;
  dists_cpu.resize(sizeof(Dist) * nsamples);  // memory was already reserved in the constructor

  Dist* dists = reinterpret_cast<Dist*>(dists_cpu.data());
  bool use_default_dist = !This().template SetupDists<Dist>(dists, nsamples);
  if (!use_default_dist) {
    dists_gpu.from_host(dists_cpu, ws.stream());
    dists = reinterpret_cast<Dist*>(dists_gpu.data());
  }

  dim3 blockDim;
  dim3 gridDim;
  blockDim.x = std::min<int>(block_sz, blockdesc_max_sz);
  blockDim.y = std::min<int>(blockdesc_count, std::max<int>(1, block_sz / blockDim.x));
  gridDim.x = 1;
  gridDim.y = div_ceil(blockdesc_count, blockDim.y);

  if (use_default_dist) {
    RNGKernel<T, Dist, true>
      <<<gridDim, blockDim, 0, ws.stream()>>>(blocks_gpu, rngs, nullptr, blockdesc_count);
  } else {
    RNGKernel<T, Dist, false>
      <<<gridDim, blockDim, 0, ws.stream()>>>(blocks_gpu, rngs, dists, blockdesc_count);
  }
}

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_RNG_BASE_GPU_CUH_
