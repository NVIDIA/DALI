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

#ifndef DALI_KERNELS_SLICE_SLICE_GPU_H_
#define DALI_KERNELS_SLICE_SLICE_GPU_H_

#include <cuda_runtime.h>
#include <vector>
#include <utility>
#include "dali/kernels/slice/slice_kernel_utils.h"
#include "dali/kernels/kernel.h"
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/dev_array.h"
#include "dali/core/error_handling.h"
#include "dali/core/cuda_error.h"
#include "dali/kernels/common/copy.h"

namespace dali {
namespace kernels {

namespace detail {

template <std::size_t Dims>
struct SliceSampleDesc {
  void *__restrict__ out;
  const void *__restrict__ in;
  DeviceArray<int64_t, Dims> in_strides;
  DeviceArray<int64_t, Dims> out_strides;
};

struct BlockDesc {
  int sampleIdx;
  std::size_t offset;
  std::size_t size;
};

template <typename OutputType, typename InputType, std::size_t Dims>
__global__ void SliceKernel(const SliceSampleDesc<Dims> *samples, const BlockDesc *blocks) {
  int sampleIdx = blocks[blockIdx.x].sampleIdx;
  size_t offset = blocks[blockIdx.x].offset + threadIdx.x;
  size_t block_end = blocks[blockIdx.x].offset + blocks[blockIdx.x].size;
  auto &sample = samples[sampleIdx];
  auto *out = static_cast<OutputType*>(sample.out);
  auto *in = static_cast<const InputType*>(sample.in);

  for (; offset < block_end; offset += blockDim.x) {
    std::size_t idx = offset;
    std::size_t out_idx = idx;
    std::size_t in_idx = 0;
    for (std::size_t d = 0; d < Dims; d++) {
      unsigned int i_d = idx / sample.out_strides[d];
      idx = idx % sample.out_strides[d];
      in_idx += i_d * sample.in_strides[d];
    }
    out[out_idx] = clamp<OutputType>(in[in_idx]);
  }
}

}  // namespace detail

template <typename OutputType, typename InputType, std::size_t Dims>
class SliceGPU {
 private:
  static constexpr std::size_t kBlockDim = 512;
  static constexpr std::size_t kBlockSize = 32 * kBlockDim;
  std::size_t block_count_ = 0;

 public:
  KernelRequirements Setup(KernelContext &context,
                           const InListGPU<InputType, Dims> &in,
                           const std::vector<SliceArgs<Dims>> &slice_args) {
    KernelRequirements req;
    ScratchpadEstimator se;
    const std::size_t num_samples = in.size();
    se.add<detail::SliceSampleDesc<Dims>>(AllocType::GPU, num_samples);

    std::vector<std::size_t> sample_sizes;
    sample_sizes.reserve(slice_args.size());
    for (auto &args : slice_args) {
      sample_sizes.push_back(volume(args.shape));
    }

    block_count_ = 0;
    for (std::size_t sample_size : sample_sizes) {
      block_count_ += std::ceil(
        sample_size / static_cast<float>(kBlockSize));
    }

    se.add<detail::BlockDesc>(AllocType::GPU, block_count_);
    req.scratch_sizes = se.sizes;

    req.output_shapes = { GetOutputShapes<Dims>(in.shape, slice_args) };
    return req;
  }

  void Run(KernelContext &context,
           OutListGPU<OutputType, Dims> &out,
           const InListGPU<InputType, Dims> &in,
           const std::vector<SliceArgs<Dims>> &slice_args) {
    const auto num_samples = in.size();

    std::vector<detail::SliceSampleDesc<Dims>> sample_descs_cpu(num_samples);
    std::vector<std::size_t> sample_sizes(in.size());
    for (int i = 0; i < in.size(); i++) {
      const auto in_shape = in.tensor_shape(i);
      const auto out_shape = out.tensor_shape(i);
      auto &sample_desc = sample_descs_cpu[i];
      sample_desc.in_strides = GetStrides<Dims>(in_shape);
      sample_desc.out_strides = GetStrides<Dims>(out_shape);
      auto &anchor = slice_args[i].anchor;
      std::size_t in_offset = 0;
      for (std::size_t d = 0; d < Dims; d++) {
        in_offset += anchor[d] * sample_desc.in_strides[d];
      }
      sample_desc.in = in.tensor_data(i) + in_offset;
      sample_desc.out = out.tensor_data(i);
      sample_sizes[i] = volume(out_shape);
    }

    detail::SliceSampleDesc<Dims> *sample_descs =
      context.scratchpad->Allocate<detail::SliceSampleDesc<Dims>>(AllocType::GPU, num_samples);
    cudaMemcpyAsync(sample_descs, sample_descs_cpu.data(),
                    num_samples * sizeof(detail::SliceSampleDesc<Dims>),
                    cudaMemcpyHostToDevice,
                    context.gpu.stream);

    std::vector<detail::BlockDesc> block_descs_cpu;
    block_descs_cpu.reserve(block_count_);
    for (int i = 0; i < num_samples; i++) {
      std::size_t offset = 0;
      std::size_t remaining = sample_sizes[i];
      while (remaining > 0) {
        std::size_t size = remaining < kBlockSize ? remaining : kBlockSize;
        block_descs_cpu.push_back({i, offset, size});
        remaining -= size;
        offset += size;
      }
    }

    detail::BlockDesc *block_descs =
      context.scratchpad->Allocate<detail::BlockDesc>(AllocType::GPU, block_count_);

    cudaMemcpyAsync(block_descs, block_descs_cpu.data(),
                    block_descs_cpu.size() * sizeof(detail::BlockDesc),
                    cudaMemcpyHostToDevice,
                    context.gpu.stream);

    const unsigned int grid = block_descs_cpu.size();
    detail::SliceKernel<OutputType, InputType, Dims>
      <<<grid, kBlockDim, 0, context.gpu.stream>>>(sample_descs, block_descs);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_GPU_H_
