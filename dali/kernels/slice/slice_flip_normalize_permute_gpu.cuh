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

#ifndef DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_GPU_H_
#define DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_GPU_H_

#include <cuda_runtime.h>
#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/cuda_error.h"
#include "dali/core/dev_array.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/common/copy.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_common.h"

namespace dali {
namespace kernels {

namespace detail {

template <size_t Dims>
struct SampleDesc {
  void *__restrict__ out;
  const void *__restrict__ in;
  DeviceArray<int64_t, Dims> in_strides;
  DeviceArray<int64_t, Dims> out_strides;
  DeviceArray<int64_t, Dims> out_shape;
  DeviceArray<int64_t, Dims> padded_out_shape;
};

struct BlockDesc {
  int sampleIdx;
  size_t offset;
  size_t size;
};

template <typename OutputType, typename InputType, unsigned Dims, bool should_normalize>
__device__ inline void SliceFlipNormalizePermuteFunc(OutputType *__restrict__ out,
                                                     const InputType *__restrict__ in,
                                                     const int64_t *out_strides,
                                                     const int64_t *in_strides,
                                                     const int64_t *out_shape,
                                                     const int64_t *padded_out_shape,
                                                     bool should_zero_pad,
                                                     unsigned norm_dim,
                                                     const float *norm_add,
                                                     const float *norm_mul,
                                                     size_t offset, size_t block_end) {
  if (Dims > 1 && !should_normalize &&
      out_strides[Dims - 1] == in_strides[Dims - 1] &&
      out_shape[Dims - 1] == padded_out_shape[Dims - 1]) {
    const unsigned NextDims = Dims > 1 ? Dims - 1 : 1;
    SliceFlipNormalizePermuteFunc<OutputType, InputType, NextDims, should_normalize>(
        out, in, out_strides, in_strides, out_shape, padded_out_shape,
        should_zero_pad, norm_dim, norm_add, norm_mul, offset, block_end);
    return;
  }

  const bool innermost_is_dense = (out_strides[Dims-1] == 1);
  for (; offset < block_end; offset += blockDim.x) {
    size_t idx = offset;
    size_t out_idx = offset;
    size_t in_idx = 0;
    unsigned norm_i = 0;
    bool zero_pad = false;

    for (unsigned d = 0; d < Dims; d++) {
      unsigned out_stride = static_cast<unsigned>(out_strides[d]);
      unsigned i_d;
      if (d == Dims-1 && innermost_is_dense) {
        i_d = idx;
        idx = 0;
      } else {
        i_d = idx / out_stride;
        idx %= out_stride;
      }
      if (zero_pad = (should_zero_pad && i_d >= out_shape[d]))
        break;

      if (d == norm_dim)
        norm_i = i_d;

      in_idx += i_d * in_strides[d];
    }

    if (zero_pad) {
      out[out_idx] = 0;
    } else {
      in_idx += idx;  // remaining dims have equal strides
      if (should_normalize) {
        float fpout = fmaf(static_cast<float>(in[in_idx]), norm_mul[norm_i], norm_add[norm_i]);
        if (std::is_integral<OutputType>::value) {
          out[out_idx] = clamp<OutputType>(__float2int_rn(fpout));
        } else {
          out[out_idx] = clamp<OutputType>(fpout);
        }
      } else {
        if (std::is_integral<OutputType>::value && std::is_floating_point<InputType>::value) {
          out[out_idx] = clamp<OutputType>(__float2int_rn(in[in_idx]));
        } else {
          out[out_idx] = clamp<OutputType>(in[in_idx]);
        }
      }
    }
  }
}

template <typename OutputType, typename InputType, size_t Dims, bool should_normalize>
__global__ void SliceFlipNormalizePermuteKernel(const SampleDesc<Dims> *samples,
                                                const BlockDesc *blocks,
                                                const float *norm_add,
                                                const float *norm_mul,
                                                unsigned normalization_dim) {
  int sampleIdx = blocks[blockIdx.x].sampleIdx;
  size_t offset = blocks[blockIdx.x].offset + threadIdx.x;
  size_t block_end = blocks[blockIdx.x].offset + blocks[blockIdx.x].size;
  auto &sample = samples[sampleIdx];
  auto *out = static_cast<OutputType *>(sample.out);
  auto *in = static_cast<const InputType *>(sample.in);

  bool should_zero_pad = false;
  for (size_t d = 0; d < Dims; d++) {
    if (should_zero_pad = (sample.padded_out_shape[d] > sample.out_shape[d])) {
      break;
    }
  }

  SliceFlipNormalizePermuteFunc<OutputType, InputType, Dims, should_normalize>(
    out, in, sample.out_strides.data(), sample.in_strides.data(),
    sample.out_shape.data(), sample.padded_out_shape.data(),
    should_zero_pad, normalization_dim, norm_add, norm_mul, offset, block_end);
}

}  // namespace detail

template <typename OutputType, typename InputType, size_t Dims>
class SliceFlipNormalizePermuteGPU {
 private:
  static constexpr size_t kBlockDim = 512;
  static constexpr size_t kBlockSize = 64 * kBlockDim;
  size_t block_count_ = 0;

 public:
  using Args = SliceFlipNormalizePermuteArgs<Dims>;
  KernelRequirements Setup(KernelContext &context,
                           const InListGPU<InputType, Dims> &in,
                           const std::vector<Args> &args) {
    KernelRequirements req;
    ScratchpadEstimator se;
    const size_t num_samples = in.size();
    se.add<detail::SampleDesc<Dims>>(AllocType::Host, num_samples);
    se.add<detail::SampleDesc<Dims>>(AllocType::GPU, num_samples);

    DALI_ENFORCE(args[0].mean.size() == args[0].inv_stddev.size());
    size_t norm_args_size = args[0].mean.size();
    if (norm_args_size > 0) {
      se.add<float>(AllocType::Host, 2 * norm_args_size);
      se.add<float>(AllocType::GPU,  2 * norm_args_size);
    }

    block_count_ = 0;
    for (auto &elem : args) {
      size_t sample_size = volume(elem.padded_shape);
      block_count_ += std::ceil(
        sample_size / static_cast<float>(kBlockSize));
    }

    se.add<detail::BlockDesc>(AllocType::Host, block_count_);
    se.add<detail::BlockDesc>(AllocType::GPU, block_count_);
    req.scratch_sizes = se.sizes;

    auto in_shapes = in.shape;
    TensorListShape<Dims> output_shapes(in_shapes.size(), Dims);
    for (int i = 0; i < in_shapes.size(); i++) {
      TensorShape<Dims> out_shape(args[i].padded_shape);
      CheckValidOutputShape<Dims>(in_shapes[i], out_shape, args[i]);
      out_shape = detail::permute<Dims>(out_shape, args[i].permuted_dims);
      output_shapes.set_tensor_shape(i, out_shape);
    }
    req.output_shapes = { output_shapes };
    return req;
  }

  void Run(KernelContext &context,
           OutListGPU<OutputType, Dims> &out,
           const InListGPU<InputType, Dims> &in,
           const std::vector<Args> &args) {
    const auto num_samples = in.size();

    auto mean_data = args[0].mean;
    auto inv_stddev_data = args[0].inv_stddev;
    DALI_ENFORCE(mean_data.size() == inv_stddev_data.size());

    detail::SampleDesc<Dims>* sample_descs_cpu =
      context.scratchpad->Allocate<detail::SampleDesc<Dims>>(AllocType::Host, num_samples);
    float *norm_add_cpu = mean_data.empty() ? nullptr :
      context.scratchpad->Allocate<float>(AllocType::Host, mean_data.size());
    float *norm_mul_cpu = inv_stddev_data.empty() ? nullptr :
      context.scratchpad->Allocate<float>(AllocType::Host, inv_stddev_data.size());
    detail::BlockDesc *block_descs_cpu =
      context.scratchpad->Allocate<detail::BlockDesc>(AllocType::Host, block_count_);

    for (size_t i = 0; i < mean_data.size(); i++) {
      norm_add_cpu[i] = -mean_data[i] * inv_stddev_data[i];
      norm_mul_cpu[i] = inv_stddev_data[i];
    }
    unsigned normalization_dim;
    std::vector<size_t> sample_sizes(in.size());
    for (int i = 0; i < in.size(); i++) {
      const auto in_shape = in.tensor_shape(i);
      auto processed_args = detail::ProcessArgs<Dims>(args[i], in_shape);
      if (i == 0) {
        normalization_dim = processed_args.normalization_dim;
      } else {
        DALI_ENFORCE(normalization_dim == processed_args.normalization_dim);
      }
      auto &sample_desc = sample_descs_cpu[i];
      sample_desc.in_strides = processed_args.in_strides;
      sample_desc.out_strides = processed_args.out_strides;
      sample_desc.out_shape = processed_args.out_shape;
      sample_desc.padded_out_shape = processed_args.padded_out_shape;
      sample_desc.in = in.tensor_data(i) + processed_args.input_offset;
      sample_desc.out = out.tensor_data(i);
      sample_sizes[i] = volume(processed_args.padded_out_shape);
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

    detail::SampleDesc<Dims> *sample_descs =
      context.scratchpad->Allocate<detail::SampleDesc<Dims>>(
        AllocType::GPU, num_samples);

    float *norm_add = mean_data.empty() ? nullptr :
      context.scratchpad->Allocate<float>(
        AllocType::GPU, mean_data.size());

    float *norm_mul = inv_stddev_data.empty() ? nullptr :
      context.scratchpad->Allocate<float>(
        AllocType::GPU, inv_stddev_data.size());

    detail::BlockDesc *block_descs =
      context.scratchpad->Allocate<detail::BlockDesc>(
        AllocType::GPU, block_count_);

    // Memory is allocated contiguously, so we launch only one cudaMemcpyAsync
    size_t total_bytes = num_samples * sizeof(detail::SampleDesc<Dims>)
      + mean_data.size() * sizeof(float)
      + inv_stddev_data.size() * sizeof(float)
      + block_count_ * sizeof(detail::BlockDesc);
    cudaMemcpyAsync(sample_descs, sample_descs_cpu,
                    total_bytes,
                    cudaMemcpyHostToDevice,
                    context.gpu.stream);

    const auto grid = block_count_;
    if (norm_add != nullptr && norm_mul != nullptr) {
      detail::SliceFlipNormalizePermuteKernel<OutputType, InputType, Dims, true>
          <<<grid, kBlockDim, 0, context.gpu.stream>>>(sample_descs, block_descs, norm_add,
                                                       norm_mul, normalization_dim);
    } else {
      detail::SliceFlipNormalizePermuteKernel<OutputType, InputType, Dims, false>
          <<<grid, kBlockDim, 0, context.gpu.stream>>>(sample_descs, block_descs, norm_add,
                                                       norm_mul, normalization_dim);
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_GPU_H_
