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

#include <cuda_runtime.h>
#include "dali/kernels/imgproc/resample/resampling_batch.h"
#include "dali/kernels/imgproc/resample/bilinear_impl.cuh"
#include "dali/kernels/imgproc/resample/nearest_impl.cuh"
#include "dali/kernels/imgproc/resample/resampling_impl.cuh"

namespace dali {
namespace kernels {

template <int which_pass, typename Output, typename Input>
__global__ void BatchedSeparableResampleKernel(
    Output *__restrict__ out,
    const Input *__restrict__ in,
    const SeparableResamplingSetup::SampleDesc *__restrict__ samples,
    const SampleBlockInfo *__restrict__ block2sample) {
  // find which part of which sample this block will process
  SampleBlockInfo sbi = block2sample[blockIdx.x];
  const SeparableResamplingSetup::SampleDesc &sample = samples[sbi.sample];
  int in_stride, out_stride;
  Output *sample_out;
  const Input *sample_in;
  int blocks;

  DeviceArray<int, 2> in_shape, out_shape;

  in_stride = sample.strides[which_pass];
  out_stride = sample.strides[which_pass+1];
  sample_in = in + sample.offsets[which_pass];
  sample_out = out + sample.offsets[which_pass+1];
  blocks = sample.block_count.pass[which_pass];
  in_shape = sample.shapes[which_pass];
  out_shape = sample.shapes[which_pass+1];

  int block = sbi.block_in_sample;

  // Axis: 0 = vertical, 1 = horizontal (HWC layout).
  int axis = which_pass;

  // If processing in VertHorz order, then the pass 0 is vertical (axis 0) and
  // pass 1 is horizontal (axis 1). If processing in HortVerz order, then axes
  // are swapped (pass 0 is axis 1 and vice versa).
  if (sample.order == SeparableResamplingSetup::HorzVert)
    axis = 1-axis;

  ResamplingFilterType ftype = sample.filter_type[axis];
  ResamplingFilter filter = sample.filter[axis];
  int support = filter.support();

  float origin = sample.origin[axis];
  float scale  = sample.scale[axis];

  int block_size = axis == 1 ? blockDim.x : blockDim.y;
  int size_in_blocks = (out_shape[axis] + block_size - 1) / block_size;

  int start = min(size_in_blocks *  block      / blocks * block_size, out_shape[axis]);
  int end   = min(size_in_blocks * (block + 1) / blocks * block_size, out_shape[axis]);

  int x0, x1, y0, y1;

  if (axis == 1) {
    x0 = start;
    x1 = end;
    y0 = 0;
    y1 = out_shape[0];
  } else {
    x0 = 0;
    x1 = out_shape[1];
    y0 = start;
    y1 = end;
  }

  switch (ftype) {
  case ResamplingFilterType::Nearest:
    if (axis == 1) {
      NNResample(x0, x1, y0, y1, origin, 0, scale, 1, sample_out, out_stride, sample_in, in_stride,
        in_shape[1], in_shape[0], sample.channels);
    } else {
      NNResample(x0, x1, y0, y1, 0, origin, 1, scale, sample_out, out_stride, sample_in, in_stride,
        in_shape[1], in_shape[0], sample.channels);
    }
    break;
  case ResamplingFilterType::Linear:
    if (axis == 1) {
      LinearHorz(x0, x1, y0, y1, origin, scale, sample_out, out_stride, sample_in, in_stride,
        in_shape[1], sample.channels);
    } else {
      LinearVert(x0, x1, y0, y1, origin, scale, sample_out, out_stride, sample_in, in_stride,
        in_shape[0], sample.channels);
    }
    break;
  default:
    if (axis == 1) {
      ResampleHorz(x0, x1, y0, y1, origin, scale, sample_out, out_stride, sample_in, in_stride,
        in_shape[1], sample.channels, filter, support);
    } else {
      ResampleVert(x0, x1, y0, y1, origin, scale, sample_out, out_stride, sample_in, in_stride,
        in_shape[0], sample.channels, filter, support);
    }
    break;
  }
}

template <int which_pass, typename Output, typename Input>
void BatchedSeparableResample(Output *out, const Input *in,
    const SeparableResamplingSetup::SampleDesc *samples,
    int num_samples, const SampleBlockInfo *block2sample, int num_blocks,
    int2 block_size,
    cudaStream_t stream) {
  if (num_blocks <= 0)
    return;

  dim3 block(block_size.x, block_size.y);

  BatchedSeparableResampleKernel<which_pass>
  <<<num_blocks, block, ResampleSharedMemSize, stream>>>(
    out, in, samples, block2sample);
}


#define INSTANTIATE_BATCHED_RESAMPLE(which_pass, Output, Input) \
template void BatchedSeparableResample<which_pass, Output, Input>( \
  Output *out, const Input *in, \
  const SeparableResamplingSetup::SampleDesc *samples, \
  int num_samples, const SampleBlockInfo *block2sample, int num_blocks, \
  int2 block_size, cudaStream_t stream)

INSTANTIATE_BATCHED_RESAMPLE(0, float, uint8_t);
INSTANTIATE_BATCHED_RESAMPLE(0, float, int8_t);
INSTANTIATE_BATCHED_RESAMPLE(0, float, uint16_t);
INSTANTIATE_BATCHED_RESAMPLE(0, float, int16_t);
INSTANTIATE_BATCHED_RESAMPLE(0, float, uint32_t);
INSTANTIATE_BATCHED_RESAMPLE(0, float, int32_t);
INSTANTIATE_BATCHED_RESAMPLE(0, float, float);

INSTANTIATE_BATCHED_RESAMPLE(1, uint8_t,  float);
INSTANTIATE_BATCHED_RESAMPLE(1, int8_t,   float);
INSTANTIATE_BATCHED_RESAMPLE(1, uint16_t, float);
INSTANTIATE_BATCHED_RESAMPLE(1, int16_t,  float);
INSTANTIATE_BATCHED_RESAMPLE(1, uint32_t, float);
INSTANTIATE_BATCHED_RESAMPLE(1, int32_t,  float);
INSTANTIATE_BATCHED_RESAMPLE(1, float,    float);

}  // namespace kernels
}  // namespace dali
