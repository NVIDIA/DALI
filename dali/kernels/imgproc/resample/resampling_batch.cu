// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>
#include "dali/kernels/imgproc/resample/resampling_batch.h"
#include "dali/kernels/imgproc/resample/bilinear_impl.cuh"
#include "dali/kernels/imgproc/resample/nearest_impl.cuh"
#include "dali/kernels/imgproc/resample/resampling_impl.cuh"

namespace dali {
namespace kernels {
namespace resampling {

template <int spatial_ndim, typename Output, typename Input>
__global__ void BatchedSeparableResampleKernel(
    int which_pass,
    const SampleDesc<spatial_ndim> *__restrict__ samples,
    const BlockDesc<spatial_ndim> *__restrict__ block2sample) {
  // find which part of which sample this block will process
  BlockDesc<spatial_ndim> bdesc = block2sample[blockIdx.x];
  const auto &sample = samples[bdesc.sample_idx];
  Output *__restrict__ sample_out;
  const Input *__restrict__ sample_in;

  ivec<spatial_ndim> in_shape;

  auto in_strides = sample.strides[which_pass];
  auto out_strides = sample.strides[which_pass+1];
  sample_in = reinterpret_cast<const Input*>(sample.pointers[which_pass]);
  sample_out = reinterpret_cast<Output*>(sample.pointers[which_pass+1]);
  in_shape = sample.shapes[which_pass];

  int axis = sample.order[which_pass];  // vec-order: 0 = X, 1 = Y, 2 = Z

  ResamplingFilterType ftype = sample.filter_type[axis];
  ResamplingFilter filter = sample.filter[axis];
  int support = filter.support();

  float origin = sample.origin[axis];
  float scale  = sample.scale[axis];

  ivec<spatial_ndim> lo = bdesc.start, hi = bdesc.end;

  switch (ftype) {
  case ResamplingFilterType::Nearest:
    {
      vec<spatial_ndim> origin_v(0.0f), scale_v(1.0f);
      origin_v[axis] = origin;
      scale_v[axis] = scale;
      NNResample(lo, hi, origin_v, scale_v,
                 sample_out, out_strides,
                 sample_in, in_strides, in_shape, sample.channels);
    }
    break;
  case ResamplingFilterType::Linear:
    if (axis == 0) {
      LinearHorz(lo, hi, origin, scale, sample_out, out_strides, sample_in,
        in_strides, in_shape, sample.channels);
    } else if (axis == 1) {
      LinearVert(lo, hi, origin, scale, sample_out, out_strides, sample_in,
        in_strides, in_shape, sample.channels);
    } else {
      LinearDepth(lo, hi, origin, scale, sample_out, out_strides, sample_in,
        in_strides, in_shape, sample.channels);
    }
    break;
  default:
    if (axis == 0) {
      ResampleHorz(lo, hi, origin, scale, sample_out, out_strides, sample_in,
        in_strides, in_shape, sample.channels, filter, support);
    } else if (axis == 1) {
      ResampleVert(lo, hi, origin, scale, sample_out, out_strides, sample_in,
        in_strides, in_shape, sample.channels, filter, support);
    } else if (axis == 2) {
      ResampleDepth(lo, hi, origin, scale, sample_out, out_strides, sample_in,
        in_strides, in_shape, sample.channels, filter, support);
    }
    break;
  }
}

template <int spatial_ndim, typename Output, typename Input>
void BatchedSeparableResample(
    int which_pass,
    const SampleDesc<spatial_ndim> *samples,
    const BlockDesc<spatial_ndim> *block2sample, int num_blocks,
    ivec3 block_size,
    int shm_size,
    cudaStream_t stream) {
  if (num_blocks <= 0)
    return;

  dim3 block(block_size.x, block_size.y, block_size.z);

  BatchedSeparableResampleKernel<spatial_ndim, Output, Input>
  <<<num_blocks, block, shm_size, stream>>>(which_pass, samples, block2sample);
  CUDA_CALL(cudaGetLastError());
}


#define INSTANTIATE_BATCHED_RESAMPLE(spatial_ndim, Output, Input)               \
template DLL_PUBLIC void BatchedSeparableResample<spatial_ndim, Output, Input>( \
  int which_pass,                                                               \
  const SampleDesc<spatial_ndim> *samples,                                      \
  const BlockDesc<spatial_ndim> *block2sample, int num_blocks,                  \
  ivec3 block_size, int shm_size, cudaStream_t stream)

// Instantiate the resampling functions.
// The resampling always goes through intermediate image of float type.
// Currently limited to only uint8 <-> float and float <-> float
// because the operator doesn't support anything else.
// To be extended when we support more image types.

INSTANTIATE_BATCHED_RESAMPLE(2, float, float);

INSTANTIATE_BATCHED_RESAMPLE(2, float, uint8_t);
INSTANTIATE_BATCHED_RESAMPLE(2, uint8_t, float);

INSTANTIATE_BATCHED_RESAMPLE(2, float, int16_t);
INSTANTIATE_BATCHED_RESAMPLE(2, int16_t, float);

INSTANTIATE_BATCHED_RESAMPLE(2, uint16_t, float);
INSTANTIATE_BATCHED_RESAMPLE(2, float, uint16_t);

INSTANTIATE_BATCHED_RESAMPLE(2, int32_t, float);
INSTANTIATE_BATCHED_RESAMPLE(2, float, int32_t);


INSTANTIATE_BATCHED_RESAMPLE(3, float, float);

INSTANTIATE_BATCHED_RESAMPLE(3, float, uint8_t);
INSTANTIATE_BATCHED_RESAMPLE(3, uint8_t, float);

INSTANTIATE_BATCHED_RESAMPLE(3, float, int16_t);
INSTANTIATE_BATCHED_RESAMPLE(3, int16_t, float);

INSTANTIATE_BATCHED_RESAMPLE(3, uint16_t, float);
INSTANTIATE_BATCHED_RESAMPLE(3, float, uint16_t);

INSTANTIATE_BATCHED_RESAMPLE(3, int32_t, float);
INSTANTIATE_BATCHED_RESAMPLE(3, float, int32_t);


}  // namespace resampling
}  // namespace kernels
}  // namespace dali
