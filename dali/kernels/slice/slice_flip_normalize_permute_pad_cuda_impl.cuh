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

#ifndef DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_KERNEL_H_
#define DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_KERNEL_H_

#include <cuda_runtime.h>
#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/cuda_error.h"
#include "dali/core/dev_array.h"
#include "dali/core/error_handling.h"
#include "dali/core/fast_div.h"
#include "dali/kernels/common/copy.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_common.h"


namespace dali {
namespace kernels {


namespace detail {

template <int Dims>
struct SampleDesc {
  void *__restrict__ out;
  const void *__restrict__ in;

  const void *__restrict__ fill_values;
  const float *__restrict__ norm_add;
  const float *__restrict__ norm_mul;
  int channel_dim;
  bool need_pad;

  TensorShape<Dims> in_strides;
  fast_div<uint64_t> out_strides[Dims];

  TensorShape<Dims> anchor;
  TensorShape<Dims> in_shape;
};

struct BlockDesc {
  int sampleIdx;
  uint64_t offset;
  uint64_t size;
};

/**
 * @brief Simplified algorithm when no padding is necessary
 * @remarks `in` already refers to the slice anchor start
 */
template <bool NeedNormalize, int Dims, typename OutputType, typename InputType>
__device__ void SliceFlipNormalizePermuteFunc(OutputType *__restrict__ out,
                                              const InputType *__restrict__ in,
                                              const fast_div<uint64_t> *out_strides,
                                              const int64_t *in_strides,
                                              const float *__restrict__ norm_add,
                                              const float *__restrict__ norm_mul, int channel_dim,
                                              uint64_t offset, uint64_t block_end) {
  if (Dims > 1 && out_strides[Dims - 1] == in_strides[Dims - 1]) {
    const int NextDims = Dims > 1 ? Dims - 1 : 1;
    SliceFlipNormalizePermuteFunc<NeedNormalize, NextDims, OutputType, InputType>(
        out, in, out_strides, in_strides, norm_add, norm_mul, channel_dim, offset, block_end);
    return;
  }

  for (; offset < block_end; offset += blockDim.x) {
    uint64_t idx = offset;
    uint64_t out_idx = idx;
    uint64_t in_idx = 0;

    int i_c = 0;
    #pragma unroll
    for (int d = 0; d < Dims; d++) {
      int i_d = div_mod(idx, idx, out_strides[d]);
      in_idx += i_d * in_strides[d];
      if (d == channel_dim)
        i_c = i_d;
    }
    in_idx += idx;  // remaining dims have equal strides
    if (NeedNormalize) {
      float fpout = fmaf(static_cast<float>(in[in_idx]), norm_mul[i_c], norm_add[i_c]);
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

DALI_HOST_DEV DALI_FORCEINLINE bool is_out_of_bounds(int64_t idx, int64_t data_extent) {
  // check idx < 0 and idx >= data_extent at once
  return static_cast<uint64_t>(idx) >= static_cast<uint64_t>(data_extent);
}


/**
 * @brief General algorithm that allows for padding in any dimension
 * @remarks `in` refers to the beginning of the input (not the slice anchor)
 * @remarks `AllDims=true` means that Dims refer to the actual number of dimensions,
 *           meaning we haven't skipped last dimensions that have same input and output strides
 */
template <bool NeedNormalize, int Dims, typename OutputType, typename InputType, bool AllDims = true>
__device__ void SliceFlipNormalizePermutePadFunc(OutputType *__restrict__ out, const InputType *__restrict__ in,
                                                 const fast_div<uint64_t> *out_strides, const int64_t *in_strides,
                                                 const int64_t *anchor, const int64_t *in_shape,
                                                 const OutputType *__restrict__ fill_values, 
                                                 const float *__restrict__ norm_add, 
                                                 const float *__restrict__ norm_mul, 
                                                 int channel_dim,
                                                 uint64_t offset, uint64_t block_end) {
  if (Dims > 1 && out_strides[Dims - 1] == in_strides[Dims - 1] && anchor[Dims - 1] == 0 &&
      channel_dim != Dims - 1) {
    const int NextDims = Dims > 1 ? Dims - 1 : 1;
    SliceFlipNormalizePermutePadFunc<NeedNormalize, NextDims, OutputType, InputType, false>(
        out, in, out_strides, in_strides, anchor, in_shape, fill_values, norm_add, norm_mul,
        channel_dim, offset, block_end);
    return;
  }

  constexpr int LastDim = Dims - 1;
  int64_t inner_in_anchor = anchor[LastDim];
  int64_t inner_in_extent = in_shape[LastDim];
  if (!AllDims) {  // if we fused dimensions, adjust inner dimension's anchor and extent
    inner_in_anchor = anchor[LastDim] * in_strides[LastDim];
    inner_in_extent = Dims > 1 ? in_strides[LastDim - 1] : in_shape[LastDim] * in_strides[LastDim];
  }

  for (; offset < block_end; offset += blockDim.x) {
    uint64_t idx = offset;
    uint64_t out_idx = idx;

    // If no dimensions were skipped (AllDims=true) we can avoid division in the last dimension,
    // because know the strides are 1 (or we treat them as 1 if we fused dimensions)
    int i_c = 0;
    int i_d;
    bool out_of_bounds = false;
    uint64_t in_idx = 0;

    #pragma unroll
    for (int d = 0; d < Dims - 1; d++) {
      i_d = div_mod(idx, idx, out_strides[d]);
      if (d == channel_dim)
        i_c = i_d;
      out_of_bounds |= is_out_of_bounds(anchor[d] + i_d, in_shape[d]);
      if (!out_of_bounds)
        in_idx += i_d * in_strides[d];
    }

    constexpr int d = LastDim;
    i_d = idx;  // out_strides[d] is 1
    if (AllDims && d == channel_dim)
      i_c = i_d;
    out_of_bounds |= is_out_of_bounds(inner_in_anchor + i_d, inner_in_extent);
    if (!out_of_bounds)
      in_idx += i_d;  // in_strides[d] is 1

    // Fill values are reused a lot, so let's make sure they are cached (by using __ldg())
    if (out_of_bounds) {
      out[out_idx] = fill_values[i_c];
    } else if (NeedNormalize) {
      float fpout = fmaf(static_cast<float>(in[in_idx]), norm_mul[i_c], norm_add[i_c]);
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

template <bool NeedPad, bool NeedNormalize, typename OutputType, typename InputType, int Dims>
__global__ void SliceFlipNormalizePermutePadKernel(const SampleDesc<Dims> *samples, const BlockDesc *blocks) {
  int sampleIdx = blocks[blockIdx.x].sampleIdx;
  uint64_t offset = blocks[blockIdx.x].offset + threadIdx.x;
  uint64_t block_end = blocks[blockIdx.x].offset + blocks[blockIdx.x].size;
  auto sample = samples[sampleIdx];
  auto *out = static_cast<OutputType*>(sample.out);
  auto *in = static_cast<const InputType*>(sample.in);
  auto *out_strides = sample.out_strides;
  auto *in_strides = sample.in_strides.data();
  auto channel_dim = sample.channel_dim;
  auto *norm_add = sample.norm_add;
  auto *norm_mul = sample.norm_mul;
  if (NeedPad && sample.need_pad) {
    auto *anchor = sample.anchor.data();
    auto *in_shape = sample.in_shape.data();
    auto *fill_values = static_cast<const OutputType*>(sample.fill_values);
    SliceFlipNormalizePermutePadFunc<NeedNormalize, Dims>(out, in, out_strides, in_strides, anchor,
                                                          in_shape, fill_values, norm_add, norm_mul,
                                                          channel_dim, offset, block_end);
  } else {
    SliceFlipNormalizePermuteFunc<NeedNormalize, Dims>(out, in, out_strides, in_strides, norm_add,
                                                       norm_mul, channel_dim, offset, block_end);
  }
}

}  // namespace detail

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_KERNEL_H_
