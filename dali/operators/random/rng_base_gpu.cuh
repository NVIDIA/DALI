// Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/operators/random/rng_base_gpu.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/batch_rng.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/dynamic_scratchpad.h"

namespace dali {
namespace rng {

namespace {  // NOLINT

template <bool value>
using bool_const = std::integral_constant<bool, value>;

template <typename T, typename Dist>
__device__ __inline__ void Generate(const SampleDesc &sample,
                                    const BlockDesc &block,
                                    Dist& dist,
                                    curandState* __restrict__ rng,
                                    bool_const<true>,     // is_noise_gen
                                    bool_const<true>) {   // is_per_channel
  auto out = static_cast<T*>(sample.output);
  auto in = static_cast<const T*>(sample.input);
  auto idx_end = block.p_offset + block.p_count;
  for (auto idx = block.p_offset + threadIdx.x; idx < idx_end; idx += blockDim.x) {
    auto n = dist.Generate(in[idx], rng);
    dist.Apply(out[idx], in[idx], n);
  }
}

template <typename T, typename Dist>
__device__ __inline__ void Generate(const SampleDesc &sample,
                                    const BlockDesc &block,
                                    Dist& dist,
                                    curandState* __restrict__ rng,
                                    bool_const<true>,     // is_noise_gen
                                    bool_const<false>) {  // is_per_channel
  auto out = static_cast<T*>(sample.output);
  auto in = static_cast<const T*>(sample.input);
  auto idx_end = block.p_offset + block.p_count;
  for (auto idx = block.p_offset + threadIdx.x; idx < idx_end; idx += blockDim.x) {
    int64_t pos = idx * sample.p_stride;
    // Implementations that generate noise once for all channels should not depend on the input
    // to generate the number.
    auto n = dist.Generate({}, rng);
    for (int c = 0; c < sample.c_count; c++, pos += sample.c_stride) {
      dist.Apply(out[pos], in[pos], n);
    }
  }
}

template <typename T, typename Dist>
__device__ __inline__ void Generate(const SampleDesc &sample,
                                    const BlockDesc &block,
                                    Dist& dist,
                                    curandState* __restrict__ rng,
                                    bool_const<false>,     // is_noise_gen
                                    bool_const<true>) {    // is_per_channel
  auto out = static_cast<T*>(sample.output);
  auto idx_end = block.p_offset + block.p_count;
  for (auto idx = block.p_offset + threadIdx.x; idx < idx_end; idx += blockDim.x) {
    auto n = dist.Generate(rng);
    out[idx] = ConvertSat<T>(n);
  }
}

template <typename T, typename Dist>
__device__ __inline__ void Generate(const SampleDesc &sample,
                                    const BlockDesc &block,
                                    Dist& dist,
                                    curandState* __restrict__ rng,
                                    bool_const<false>,      // is_noise_gen
                                    bool_const<false>) {    // is_per_channel
  auto out = static_cast<T*>(sample.output);
  auto idx_end = block.p_offset + block.p_count;
  for (auto idx = block.p_offset + threadIdx.x; idx < idx_end; idx += blockDim.x) {
    int64_t pos = idx * sample.p_stride;
    auto n = dist.Generate(rng);
    for (int c = 0; c < sample.c_count; c++, pos += sample.c_stride) {
      out[pos] = ConvertSat<T>(n);
    }
  }
}

template <typename T, typename Dist, bool DefaultDist, bool IsNoiseGen, bool IsPerChannel>
__global__ void RNGKernel(SampleDesc* __restrict__ sample_descs,
                          BlockDesc* __restrict__ block_descs,
                          curandState* __restrict__ states,
                          const Dist* __restrict__ dists, int nblocks) {
  int block_size = blockDim.x * blockDim.y;
  int local_tid = blockDim.x * threadIdx.y + threadIdx.x;
  int64_t global_tid = blockIdx.y * block_size + local_tid;
  auto rng = states + global_tid;
  int blk_stride = blockDim.y * gridDim.y;
  int blk = blockIdx.y * blockDim.y + threadIdx.y;
  for (; blk < nblocks; blk += blk_stride) {
    auto block = block_descs[blk];
    auto sample = sample_descs[block.sample_idx];
    Dist dist = DefaultDist ? Dist() : dists[block.sample_idx];
    Generate<T, Dist>(sample, block, dist, rng,
                      bool_const<IsNoiseGen>(), bool_const<IsPerChannel>());
  }
}

}  // namespace

template <typename Backend, typename Impl, bool IsNoiseGen>
template <typename T, typename Dist>
void RNGBase<Backend, Impl, IsNoiseGen>::RunImplTyped(Workspace &ws, GPUBackend) {
  static_assert(std::is_same<Backend, GPUBackend>::value, "Unexpected backend");
  auto &output = ws.Output<GPUBackend>(0);
  auto rngs = backend_data_.randomizer_.states();
  int block_sz = backend_data_.block_size_;
  int max_nblocks = backend_data_.max_blocks_;
  int blockdesc_count = -1;
  TensorListView<StorageGPU, const T> in_view;
  auto out_view = view<T>(output);
  int nsamples = out_view.num_samples();
  if (nsamples == 0) {
    return;
  }

  if (IsNoiseGen) {
    const auto& input = ws.Input<GPUBackend>(0);
    in_view = view<const T>(input);
    output.SetLayout(input.GetLayout());
  }

  // TODO(janton): set layout explicitly from the user for RNG
  auto layout = output.GetLayout();
  bool independent_channels = This().PerChannel();

  // Channel dimension is specified only when we follow the "generate once, apply to
  // all channels" aproach
  int channel_dim = -1;
  if (!independent_channels) {
    channel_dim = layout.empty() ? out_view.shape.sample_dim() - 1 : layout.find('C');
  }

  auto &samples_cpu = backend_data_.sample_descs_cpu_;
  samples_cpu.resize(nsamples);
  SetupSampleDescs(samples_cpu.data(), out_view, in_view, channel_dim);

  auto &blocks_cpu = backend_data_.block_descs_cpu_;
  blocks_cpu.resize(max_nblocks);
  blockdesc_count =
      SetupBlockDescs(blocks_cpu.data(), block_sz, max_nblocks, out_view.shape, channel_dim);
  if (blockdesc_count == 0) {
    return;
  }
  blocks_cpu.resize(blockdesc_count);

  int64_t blockdesc_max_sz = -1;
  for (int b = 0; b < blockdesc_count; b++) {
    int64_t block_sz = blocks_cpu[b].p_count;
    if (block_sz > blockdesc_max_sz)
      blockdesc_max_sz = block_sz;
  }

  kernels::DynamicScratchpad scratch(ws.stream());

  auto dists_cpu = make_span(scratch.Allocate<mm::memory_kind::host, Dist>(nsamples), nsamples);
  bool use_default_dist = !This().template SetupDists<T>(dists_cpu.data(), ws, nsamples);

  rng::SampleDesc* samples_gpu = nullptr;
  rng::BlockDesc* blocks_gpu = nullptr;
  Dist* dists_gpu = nullptr;
  if (!use_default_dist) {
    std::tie(samples_gpu, blocks_gpu, dists_gpu) =
      scratch.ToContiguousGPU(ws.stream(), samples_cpu, blocks_cpu, dists_cpu);
  } else {
    std::tie(samples_gpu, blocks_gpu) =
      scratch.ToContiguousGPU(ws.stream(), samples_cpu, blocks_cpu);
  }

  dim3 blockDim;
  dim3 gridDim;
  blockDim.x = std::min<int>(block_sz, blockdesc_max_sz);
  blockDim.y = std::min<int>(blockdesc_count, std::max<int>(1, block_sz / blockDim.x));
  gridDim.x = 1;
  gridDim.y = div_ceil(blockdesc_count, blockDim.y);

  VALUE_SWITCH(use_default_dist ? 1 : 0, DefaultDist, (false, true), (
    VALUE_SWITCH(independent_channels ? 1 : 0, IsPerChannel, (false, true), (
      RNGKernel<T, Dist, DefaultDist, IsNoiseGen, IsPerChannel>
        <<<gridDim, blockDim, 0, ws.stream()>>>(samples_gpu, blocks_gpu,
                                                rngs, dists_gpu, blockdesc_count);
    ), ());  // NOLINT
  ), ());  // NOLINT
  CUDA_CALL(cudaGetLastError());
}

}  // namespace rng
}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_RNG_BASE_GPU_CUH_
