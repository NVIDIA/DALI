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

template <int Dims>
struct SliceSampleDesc {
  void *__restrict__ out;
  const void *__restrict__ in;
  TensorShape<Dims> in_strides;
  TensorShape<Dims> out_strides;
};

struct BlockDesc {
  int sampleIdx;
  size_t offset;
  size_t size;
};

template <int Dims, typename OutputType, typename InputType>
__device__ void SliceFunc(OutputType *__restrict__ out, const InputType *__restrict__ in,
                          const int64_t *out_strides, const int64_t *in_strides,
                          size_t offset, size_t block_end) {
  if (Dims > 1 && out_strides[Dims-1] == in_strides[Dims-1]) {
    const int NextDims = Dims > 1 ? Dims-1 : 1;
    SliceFunc<NextDims>(out, in, out_strides, in_strides, offset, block_end);
    return;
  }

  for (; offset < block_end; offset += blockDim.x) {
    size_t idx = offset;
    size_t out_idx = idx;
    size_t in_idx = 0;
    for (int d = 0; d < Dims; d++) {
      int i_d = idx / out_strides[d];
      idx %= static_cast<size_t>(out_strides[d]);
      in_idx += i_d * static_cast<size_t>(in_strides[d]);
    }
    in_idx += idx;  // remaining dims have equal strides
    out[out_idx] = clamp<OutputType>(in[in_idx]);
  }
}

template <typename OutputType, typename InputType, int Dims>
__global__ void SliceKernel(const SliceSampleDesc<Dims> *samples, const BlockDesc *blocks) {
  int sampleIdx = blocks[blockIdx.x].sampleIdx;
  size_t offset = blocks[blockIdx.x].offset + threadIdx.x;
  size_t block_end = blocks[blockIdx.x].offset + blocks[blockIdx.x].size;
  auto sample = samples[sampleIdx];
  auto *out = static_cast<OutputType*>(sample.out);
  auto *in = static_cast<const InputType*>(sample.in);
  SliceFunc<Dims>(out, in, sample.out_strides.data(), sample.in_strides.data(), offset, block_end);
}

}  // namespace detail

template <typename OutputType, typename InputType, int Dims>
class SliceGPU {
 private:
  static constexpr size_t kBlockDim = 256;
  static constexpr size_t kBlockSize = 64 * kBlockDim;
  size_t block_count_ = 0;

 public:
  KernelRequirements Setup(KernelContext &context,
                           const InListGPU<InputType, Dims> &in,
                           const std::vector<SliceArgs<Dims>> &slice_args) {
    KernelRequirements req;
    ScratchpadEstimator se;
    const size_t num_samples = in.size();
    se.add<detail::SliceSampleDesc<Dims>>(AllocType::Host, num_samples);
    se.add<detail::SliceSampleDesc<Dims>>(AllocType::GPU, num_samples);

    std::vector<size_t> sample_sizes;
    sample_sizes.reserve(slice_args.size());
    for (auto &args : slice_args) {
      sample_sizes.push_back(volume(args.shape));
    }

    block_count_ = 0;
    for (size_t sample_size : sample_sizes) {
      block_count_ += std::ceil(
        sample_size / static_cast<float>(kBlockSize));
    }

    se.add<detail::BlockDesc>(AllocType::Host, block_count_);
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

    detail::SliceSampleDesc<Dims>* sample_descs_cpu =
      context.scratchpad->Allocate<detail::SliceSampleDesc<Dims>>(AllocType::Host, num_samples);
    detail::BlockDesc *block_descs_cpu =
      context.scratchpad->Allocate<detail::BlockDesc>(AllocType::Host, block_count_);

    std::vector<size_t> sample_sizes(in.size());
    for (int i = 0; i < in.size(); i++) {
      const auto in_shape = in.tensor_shape(i);
      const auto out_shape = out.tensor_shape(i);
      auto &sample_desc = sample_descs_cpu[i];
      sample_desc.in_strides = GetStrides(in_shape);
      sample_desc.out_strides = GetStrides(out_shape);
      auto &anchor = slice_args[i].anchor;
      size_t in_offset = 0;
      for (int d = 0; d < Dims; d++) {
        in_offset += anchor[d] * sample_desc.in_strides[d];
      }
      sample_desc.in = in.tensor_data(i) + in_offset;
      sample_desc.out = out.tensor_data(i);
      sample_sizes[i] = volume(out_shape);
    }

    size_t block_idx = 0;
    for (int i = 0; i < num_samples; i++) {
      size_t offset = 0;
      size_t remaining = sample_sizes[i];
      while (remaining > 0) {
        size_t size = remaining < kBlockSize ? remaining : kBlockSize;
        block_descs_cpu[block_idx++] = {i, offset, size};
        remaining -= size;
        offset += size;
      }
    }

    detail::SliceSampleDesc<Dims> *sample_descs =
      context.scratchpad->Allocate<detail::SliceSampleDesc<Dims>>(
        AllocType::GPU, num_samples);
    detail::BlockDesc *block_descs =
      context.scratchpad->Allocate<detail::BlockDesc>(
        AllocType::GPU, block_count_);

    // Memory is allocated contiguously, so we launch only one cudaMemcpyAsync
    size_t total_bytes = num_samples * sizeof(detail::SliceSampleDesc<Dims>)
      + block_count_ * sizeof(detail::BlockDesc);
    cudaMemcpyAsync(sample_descs, sample_descs_cpu,
                    total_bytes,
                    cudaMemcpyHostToDevice,
                    context.gpu.stream);

    const auto grid = block_count_;
    detail::SliceKernel<OutputType, InputType, Dims>
      <<<grid, kBlockDim, 0, context.gpu.stream>>>(sample_descs, block_descs);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_GPU_H_
