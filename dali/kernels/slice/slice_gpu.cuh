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

  const void *__restrict__ fill_values;
  int channel_dim;

  TensorShape<Dims> in_strides;
  TensorShape<Dims> out_strides;

  TensorShape<Dims> anchor;
  TensorShape<Dims> in_shape;
};

struct BlockDesc {
  int sampleIdx;
  int64_t offset;
  int64_t size;
};

template <int Dims, typename OutputType, typename InputType>
__device__ void SliceFunc(OutputType *__restrict__ out, const InputType *__restrict__ in,
                          const int64_t *out_strides, const int64_t *in_strides,
                          const int64_t *anchor, const int64_t *in_shape,
                          const OutputType *__restrict__ fill_values, int channel_dim,
                          size_t offset, size_t block_end) {
  if (Dims > 1 && out_strides[Dims - 1] == in_strides[Dims - 1] && anchor[Dims - 1] == 0 && false) {  // TODO(janton): fix
    const int NextDims = Dims > 1 ? Dims - 1 : 1;
    SliceFunc<NextDims>(out, in, out_strides, in_strides, anchor, in_shape, fill_values,
                        channel_dim, offset, block_end);
    return;
  }

  for (; offset < block_end; offset += blockDim.x) {
    int64_t idx = offset;
    int64_t out_idx = idx;
    int64_t in_idx = 0;
    int i_c = 0;
    bool out_of_range = false;
    for (int d = 0; d < Dims; d++) {
      int i_d = idx / out_strides[d];
      idx %= out_strides[d];
      if (d == channel_dim)
        i_c = i_d;

      int in_i_d = anchor[d] + i_d;
      out_of_range |= in_i_d < 0 || in_i_d >= in_shape[d];
      if (!out_of_range)
        in_idx += in_i_d * in_strides[d];
    }
    in_idx += idx;  // remaining dims have equal strides
    out[out_idx] = out_of_range ? fill_values[i_c] : clamp<OutputType>(in[in_idx]);
  }
}

template <typename OutputType, typename InputType, int Dims>
__global__ void SliceKernel(const SliceSampleDesc<Dims> *samples, const BlockDesc *blocks) {
  int sampleIdx = blocks[blockIdx.x].sampleIdx;
  int64_t offset = blocks[blockIdx.x].offset + threadIdx.x;
  int64_t block_end = blocks[blockIdx.x].offset + blocks[blockIdx.x].size;
  auto sample = samples[sampleIdx];
  auto *out = static_cast<OutputType*>(sample.out);
  auto *in = static_cast<const InputType*>(sample.in);
  auto *fill_values = static_cast<const OutputType*>(sample.fill_values);
  SliceFunc<Dims>(out, in, sample.out_strides.data(), sample.in_strides.data(),
                  sample.anchor.data(), sample.in_shape.data(), fill_values, sample.channel_dim,
                  offset, block_end);
}

}  // namespace detail

template <typename OutputType, typename InputType, int Dims>
class SliceGPU {
 private:
  static constexpr int64_t kBlockDim = 256;
  static constexpr int64_t kBlockSize = 64 * kBlockDim;
  int64_t block_count_ = 0;

 public:
  KernelRequirements Setup(KernelContext &context,
                           const InListGPU<InputType, Dims> &in,
                           const std::vector<SliceArgs<OutputType, Dims>> &slice_args) {
    KernelRequirements req;
    ScratchpadEstimator se;
    auto num_samples = in.size();
    se.add<detail::SliceSampleDesc<Dims>>(AllocType::Host, num_samples);
    se.add<detail::SliceSampleDesc<Dims>>(AllocType::GPU, num_samples);

    nfill_values_ = 0;
    for (const auto& args: slice_args) {
      if (nfill_values_ == 0)
        nfill_values_ = args.fill_values.size();
      else
        DALI_ENFORCE(nfill_values_ == args.fill_values.size(),
          "The number of fill values should be the same for all the samples");
    }
    if (nfill_values_ == 0) {
      default_fill_values_ = true;
      nfill_values_ = 1;
    } else if (nfill_values_ > 1) {
      for (const auto& args: slice_args) {
        DALI_ENFORCE(args.channel_dim >= 0 && args.channel_dim < Dims,
          "Channel dim must be valid for multi-channel fill values");
        DALI_ENFORCE(nfill_values_ == args.shape[args.channel_dim],
          "The number of fill values should match the number of channels in the output slice");
      }
    }

    se.add<OutputType>(AllocType::Host, num_samples * nfill_values_);
    se.add<OutputType>(AllocType::GPU, num_samples * nfill_values_);

    std::vector<int64_t> sample_sizes;
    sample_sizes.reserve(slice_args.size());
    for (auto &args : slice_args) {
      sample_sizes.push_back(volume(args.shape));
    }

    block_count_ = 0;
    for (auto sample_size : sample_sizes) {
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
           const std::vector<SliceArgs<OutputType, Dims>> &slice_args) {
    const auto num_samples = in.size();

    // Host memory
    detail::SliceSampleDesc<Dims>* sample_descs_cpu =
      context.scratchpad->Allocate<detail::SliceSampleDesc<Dims>>(AllocType::Host, num_samples);
    OutputType *fill_values_cpu =
        context.scratchpad->Allocate<OutputType>(AllocType::Host, num_samples * nfill_values_);
    detail::BlockDesc *block_descs_cpu =
      context.scratchpad->Allocate<detail::BlockDesc>(AllocType::Host, block_count_);

    // GPU memory
    detail::SliceSampleDesc<Dims> *sample_descs =
      context.scratchpad->Allocate<detail::SliceSampleDesc<Dims>>(
        AllocType::GPU, num_samples);
    OutputType *fill_values_gpu =
        context.scratchpad->Allocate<OutputType>(AllocType::GPU, num_samples * nfill_values_);
    detail::BlockDesc *block_descs =
      context.scratchpad->Allocate<detail::BlockDesc>(
        AllocType::GPU, block_count_);

    std::vector<int64_t> sample_sizes(in.size());
    for (int i = 0; i < in.size(); i++) {
      const auto in_shape = in.tensor_shape(i);
      const auto out_shape = out.tensor_shape(i);
      const auto anchor = slice_args[i].anchor;
      auto &sample_desc = sample_descs_cpu[i];
      sample_desc.in_strides = GetStrides(in_shape);
      sample_desc.out_strides = GetStrides(out_shape);
      sample_desc.anchor = anchor;
      sample_desc.in_shape = in_shape;
      sample_desc.in = in.tensor_data(i);
      sample_desc.out = out.tensor_data(i);
      sample_sizes[i] = volume(out_shape);

      // We are filling fill_values_cpu but the sample desc will point to GPU memory
      if (default_fill_values_) {
        assert(nfill_values_ == 1);
        fill_values_cpu[i] = OutputType(0);
      } else {
        auto *fill_values = fill_values_cpu + i * nfill_values_;
        for (int d = 0; d < nfill_values_; d++)
          fill_values[d] = slice_args[i].fill_values[d];
      }
      sample_desc.fill_values = fill_values_gpu + i * nfill_values_;
      sample_desc.channel_dim = nfill_values_ > 1 ? slice_args[i].channel_dim : -1;
    }

    int64_t block_idx = 0;
    for (int i = 0; i < num_samples; i++) {
      int64_t offset = 0;
      int64_t remaining = sample_sizes[i];
      while (remaining > 0) {
        int64_t size = remaining < kBlockSize ? remaining : kBlockSize;
        block_descs_cpu[block_idx++] = {i, offset, size};
        remaining -= size;
        offset += size;
      }
    }

    // Memory is allocated contiguously, so we launch only one cudaMemcpyAsync
    int64_t total_bytes = num_samples * sizeof(detail::SliceSampleDesc<Dims>)
        + num_samples * nfill_values_ * sizeof(OutputType)
        + block_count_ * sizeof(detail::BlockDesc);
    cudaMemcpyAsync(sample_descs, sample_descs_cpu, total_bytes, cudaMemcpyHostToDevice,
                    context.gpu.stream);

    const auto grid = block_count_;
    detail::SliceKernel<OutputType, InputType, Dims>
      <<<grid, kBlockDim, 0, context.gpu.stream>>>(sample_descs, block_descs);
  }
 private:
  int nfill_values_ = 0;
  bool default_fill_values_ = false;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_GPU_H_
