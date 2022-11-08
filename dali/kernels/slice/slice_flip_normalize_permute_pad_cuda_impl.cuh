// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_CUDA_IMPL_CUH_
#define DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_CUDA_IMPL_CUH_

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

namespace slice_impl {

template <int Dims>
struct SampleDesc {
  void *__restrict__ out;
  const void *__restrict__ in;

  TensorShape<Dims> out_shape;
  TensorShape<Dims> in_shape;
  TensorShape<Dims> anchor;

  const void *__restrict__ fill_values;
  const float *__restrict__ norm_add;
  const float *__restrict__ norm_mul;
  int channel_dim;
  bool need_pad;
  bool need_flip;
  int effective_ndim;

  fast_div<uint64_t> out_strides[Dims];
  TensorShape<Dims> in_strides;
};

struct BlockDesc {
  int sampleIdx;
  uint64_t offset;
  uint64_t size;
};

DALI_HOST_DEV DALI_FORCEINLINE bool is_out_of_bounds(int64_t idx, int64_t data_extent) {
  // check idx < 0 and idx >= data_extent at once
  return static_cast<uint64_t>(idx) >= static_cast<uint64_t>(data_extent);
}

/**
 * @brief General algorithm that allows for padding in any dimension
 * @remarks `in` refers to the slice anchor start
 */
template <bool NeedFlip, bool NeedNormalize, bool NeedPad, int Dims,
          typename Out, typename In, bool AllDims = true>
__device__ void SliceFlipNormalizePermutePadFunc(
    Out *__restrict__ out, const In *__restrict__ in,
    const fast_div<uint64_t> *out_strides, const int64_t *in_strides, const int64_t *out_shape,
    const int64_t *in_shape, const int64_t *anchor, const Out *__restrict__ fill_values,
    const float *__restrict__ norm_add, const float *__restrict__ norm_mul, int channel_dim,
    int effective_ndim, uint64_t offset, uint64_t block_end) {
  if (Dims > effective_ndim) {
    const int NextDims = Dims > 1 ? Dims - 1 : 1;
    SliceFlipNormalizePermutePadFunc<NeedFlip, NeedNormalize, NeedPad, NextDims, Out, In, false>(
        out, in, out_strides, in_strides, out_shape, in_shape, anchor, fill_values, norm_add,
        norm_mul, channel_dim, effective_ndim, offset, block_end);
    return;
  }

  for (; offset < block_end; offset += blockDim.x) {
    uint64_t idx = offset;
    uint64_t out_idx = idx;

    // We can avoid division in the last dimension because know the strides are 1
    // (or we treat them as 1 if we fused dimensions)
    int i_c = 0;
    int i_d;
    bool out_of_bounds = false;
    uint64_t in_idx = 0;

    #pragma unroll
    for (int d = 0; d < Dims - 1; d++) {
      i_d = div_mod(idx, idx, out_strides[d]);
      if (d == channel_dim)
        i_c = i_d;
      if (NeedPad) {
        auto in_i_d = NeedFlip && in_strides[d] < 0 ? anchor[d] + out_shape[d] - 1 - i_d
                                                    : anchor[d] + i_d;
        out_of_bounds |= is_out_of_bounds(in_i_d, in_shape[d]);
      }
      in_idx += i_d * in_strides[d];
    }

    constexpr int d = Dims - 1;
    i_d = idx;  // out_strides[d] is treated as 1
    if (AllDims && d == channel_dim)
      i_c = i_d;

    if (NeedPad) {
      auto in_i_d = NeedFlip && in_strides[d] < 0 ? anchor[d] + out_shape[d] - 1 - i_d
                                                  : anchor[d] + i_d;
      out_of_bounds |= is_out_of_bounds(in_i_d, in_shape[d]);
    }

    if (AllDims) {
      in_idx += i_d * in_strides[d];
    } else {
      // abs(in_strides[d]) is 1 but we care about the sign
      in_idx += in_strides[d] < 0 ? -i_d : i_d;
    }

    if (NeedPad && out_of_bounds) {
      out[out_idx] = fill_values[i_c];
    } else if (NeedNormalize) {
      float fpout = fmaf(static_cast<float>(in[in_idx]), norm_mul[i_c], norm_add[i_c]);
      out[out_idx] = ConvertSat<Out>(fpout);
    } else {
      out[out_idx] = ConvertSat<Out>(in[in_idx]);
    }
  }
}

template <bool NeedPad, bool NeedFlip, bool NeedNormalize, typename Out, typename In, int Dims>
__global__ void SliceFlipNormalizePermutePadKernel(const SampleDesc<Dims> *samples,
                                                   const BlockDesc *blocks) {
  int sampleIdx = blocks[blockIdx.x].sampleIdx;
  uint64_t offset = blocks[blockIdx.x].offset + threadIdx.x;
  uint64_t block_end = blocks[blockIdx.x].offset + blocks[blockIdx.x].size;
  auto sample = samples[sampleIdx];
  auto *out = static_cast<Out*>(sample.out);
  auto *in = static_cast<const In*>(sample.in);
  auto *out_strides = sample.out_strides;
  auto *in_strides = sample.in_strides.data();
  auto *out_shape = sample.out_shape.data();
  auto *in_shape = sample.in_shape.data();
  auto *anchor = sample.anchor.data();
  auto *fill_values = static_cast<const Out*>(sample.fill_values);
  VALUE_SWITCH(NeedPad && sample.need_pad, SampleNeedPad, (false, true), (
    VALUE_SWITCH(NeedFlip && sample.need_flip, SampleNeedFlip, (false, true), (
      SliceFlipNormalizePermutePadFunc<SampleNeedFlip, NeedNormalize, SampleNeedPad, Dims>(
          out, in, out_strides, in_strides, out_shape, in_shape, anchor, fill_values,
          sample.norm_add, sample.norm_mul, sample.channel_dim, sample.effective_ndim,
          offset, block_end);
    ), ());  // NOLINT
  ), ());  // NOLINT
}

}  // namespace slice_impl

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_CUDA_IMPL_CUH_
