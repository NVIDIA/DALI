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
  TensorShape<Dims> in_strides;
  TensorShape<Dims> out_strides;
  TensorShape<Dims> out_shape;
  TensorShape<Dims> padded_out_shape;
  float padding_val;
};

struct BlockDesc {
  int sampleIdx;
  size_t offset;
  size_t size;
};

template <typename OutputType, typename InputType, unsigned Dims, bool should_normalize>
__device__ inline void SliceFlipNormalizePermutePadFunc(OutputType *__restrict__ out,
                                                        const InputType *__restrict__ in,
                                                        const int64_t *out_strides,
                                                        const int64_t *in_strides,
                                                        const int64_t *out_shape,
                                                        const int64_t *padded_out_shape,
                                                        bool should_pad,
                                                        unsigned norm_dim,
                                                        const float *norm_add,
                                                        const float *norm_mul,
                                                        OutputType padding_val,
                                                        size_t offset, size_t block_end) {
  if (Dims > 1 && !should_normalize &&
      out_strides[Dims - 1] == in_strides[Dims - 1] &&
      out_shape[Dims - 1] == padded_out_shape[Dims - 1]) {
    const unsigned NextDims = Dims > 1 ? Dims - 1 : 1;
    SliceFlipNormalizePermutePadFunc<OutputType, InputType, NextDims, should_normalize>(
        out, in, out_strides, in_strides, out_shape, padded_out_shape,
        should_pad, norm_dim, norm_add, norm_mul, padding_val, offset, block_end);
    return;
  }

  const bool innermost_is_dense = (out_strides[Dims-1] == 1);
  for (; offset < block_end; offset += blockDim.x) {
    size_t idx = offset;
    size_t out_idx = offset;
    size_t in_idx = 0;
    unsigned norm_i = 0;
    bool pad = false;

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
      if (pad = (should_pad && i_d >= out_shape[d]))
        break;

      if (d == norm_dim)
        norm_i = i_d;

      in_idx += i_d * in_strides[d];
    }

    if (pad) {
      out[out_idx] = padding_val;
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

template <typename OutputType, typename InputType, int Dims, bool should_normalize>
__global__ void SliceFlipNormalizePermutePadKernel(const SampleDesc<Dims> *samples,
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

  bool should_pad = false;
  for (int d = 0; d < Dims; d++) {
    if (should_pad = (sample.padded_out_shape[d] > sample.out_shape[d])) {
      break;
    }
  }

  SliceFlipNormalizePermutePadFunc<OutputType, InputType, Dims, should_normalize>(
    out, in, sample.out_strides.data(), sample.in_strides.data(),
    sample.out_shape.data(), sample.padded_out_shape.data(),
    should_pad, normalization_dim, norm_add, norm_mul,
    Convert<OutputType>(sample.padding_val), offset, block_end);
}

}  // namespace detail

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_KERNEL_H_
