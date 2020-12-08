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

template <>
struct RNGBaseFields<GPUBackend> {
  RNGBaseFields<GPUBackend>(int64_t seed, int max_batch_size)
      : max_blocks_(std::max(max_batch_size, 1024)),
        randomizer_(seed, block_size_ * max_blocks_) {
    block_descs_gpu_ =
        kernels::memory::alloc_unique<BlockDesc>(kernels::AllocType::GPU, max_blocks_);
    block_descs_cpu_ =
        kernels::memory::alloc_unique<BlockDesc>(kernels::AllocType::Pinned, max_blocks_);
  }
  static constexpr int block_size_ = 256;
  const int max_blocks_;
  curand_states randomizer_;
  kernels::memory::KernelUniquePtr<BlockDesc> block_descs_gpu_;
  kernels::memory::KernelUniquePtr<BlockDesc> block_descs_cpu_;
};

namespace {

template <typename Out, typename Dist, bool DefaultDist = false>
__global__ void RNGKernel(BlockDesc *block_descs, curandState* states, Dist* dists) {
  auto desc = block_descs[blockIdx.x];
  auto start = static_cast<Out*>(desc.start);
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto block_end = start + desc.size;
  Dist default_dist;
  Dist& dist = DefaultDist ? default_dist : dists[desc.sample_idx];
  for (auto out = start + threadIdx.x; out < block_end; out += blockDim.x) {
    *out = ConvertSat<Out>(dist.yield(states + tid));
  }
}

template <typename Out, typename Dist, bool DefaultDist = false>
__global__ void RNGKernelSingleValue(BlockDesc *descs, curandState* states, Dist* dists,
                                     int nsamples) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nsamples)
    return;
  auto desc = descs[tid];
  auto out = static_cast<Out*>(desc.start);
  Dist default_dist;
  Dist& dist = DefaultDist ? default_dist : dists[desc.sample_idx];
  *out = ConvertSat<Out>(dist.yield(states + tid));
}

}  // namespace

template <typename Backend, typename Impl>
template <typename T, typename Dist>
void RNGBase<Backend, Impl>::RunImplTyped(workspace_t<GPUBackend> &ws) {
  static_assert(std::is_same<Backend, GPUBackend>::value);
  auto out_view = view<T>(ws.template OutputRef<GPUBackend>(0));
  int nsamples = out_view.shape.size();
  auto blocks_cpu = backend_specific_.block_descs_cpu_.get();
  auto blocks_gpu = backend_specific_.block_descs_gpu_.get();
  auto rngs = backend_specific_.randomizer_.states();
  int64_t block_sz = backend_specific_.block_size_;
  int64_t max_nblocks = backend_specific_.max_blocks_;
  int64_t nblocks = -1;
  if (single_value_) {
    nblocks = SetupBlockDescsSingleValue(
      blocks_cpu, max_nblocks, out_view, ws.stream());
  } else {
    nblocks = SetupBlockDescs(
      blocks_cpu, block_sz, max_nblocks, out_view, ws.stream());
  }
  cudaMemcpyAsync(blocks_gpu, blocks_cpu,
                  sizeof(BlockDesc) * nblocks, cudaMemcpyHostToDevice, ws.stream());

  Dist* dists = This().template SetupDists<Dist>(nsamples, ws.stream());

  constexpr bool use_default = !Impl::template Dist<T>::has_state;
  if (single_value_) {
    RNGKernelSingleValue<T, Dist, use_default>
      <<<nblocks, block_sz, 0, ws.stream()>>>(blocks_gpu, rngs, dists, nsamples);
  } else {
    RNGKernel<T, Dist, use_default>
      <<<nblocks, block_sz, 0, ws.stream()>>>(blocks_gpu, rngs, dists);
  }
}

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_RNG_BASE_GPU_CUH_
