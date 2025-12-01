// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <utility>
#include <vector>
#include "dali/core/format.h"
#include "dali/operators/bbox/bb_flip.cuh"
#include "dali/kernels/dynamic_scratchpad.h"

namespace dali {

/**
 * @param samples - Sample description (input/output pointer + flipping configuration)
 * @param blocks  - Mapping the current CUDA block to range within particular sample
 */
template <bool ltrb>
__global__ void BbFlipKernel(const BbFlipSampleDesc *samples, const kernels::BlockDesc<1> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];

  for (int idx = threadIdx.x + block.start.x; idx < block.end.x; idx += blockDim.x) {
    bool h = sample.horz;
    bool v = sample.vert;

    const auto *in = &sample.input[4 * idx];
    auto *out = &sample.output[4 * idx];
    if (ltrb) {
      out[0] = h ? 1.0f - in[2] : in[0];
      out[1] = v ? 1.0f - in[3] : in[1];
      out[2] = h ? 1.0f - in[0] : in[2];
      out[3] = v ? 1.0f - in[1] : in[3];
    } else {
      // No range checking required if the parenthesis is respected in the two lines below.
      // If the original bounding box satisfies the condition that x + w <= 1.0f, then the
      // expression 1.0f - (x + w) is guaranteed to yield a non-negative result. QED.
      out[0] = h ? 1.0f - (in[0] + in[2]) : in[0];
      out[1] = v ? 1.0f - (in[1] + in[3]) : in[1];
      out[2] = in[2];  // width and
      out[3] = in[3];  // height remain unaffected
    }
  }
}

TensorListShape<2> GetNormalizedShape(const TensorListShape<-1> &shape) {
  if (shape.sample_dim() == 2) {
    return shape.to_static<2>();
  }
  if (shape.sample_dim() > 2) {
    std::array<std::pair<int, int>, 1> collapse_group = {{{0, shape.sample_dim() - 1}}};
    return collapse_dims<2>(shape, collapse_group);
  }
  TensorListShape<2> result(shape.num_samples(), 2);
  for (int i = 0; i < shape.num_samples(); i++) {
    auto tspan = shape.tensor_shape_span(i);
    result.set_tensor_shape(i, {tspan[0] / 4, 4});
  }
  return result;
}

void BbFlipGPU::RunImpl(Workspace &ws) {
  auto &input = ws.Input<GPUBackend>(0);
  const auto &shape = input.shape();
  auto nsamples = shape.num_samples();
  auto &output = ws.Output<GPUBackend>(0);

  DALI_ENFORCE(IsType<float>(input.type()),
               make_string("Expected input data as float; got ", input.type()));
  DALI_ENFORCE(input._num_elements() % 4 == 0,
               make_string("Input data size must be a multiple of 4 if it contains bounding",
                           " boxes;  got ", input._num_elements()));


  for (int sample = 0; sample < nsamples; sample++) {
    auto dim = shape[sample].sample_dim();

    DALI_ENFORCE(dim < 2 || shape[sample][dim - 1] == 4,
                 "If bounding box tensor is >= 2D, innermost dimension must be 4");
    DALI_ENFORCE(dim > 1 || shape[sample][0] % 4 == 0,
                 "Flat representation of bounding boxes must have size divisible by 4");
  }

  TensorListShape<2> strong_shape = GetNormalizedShape(shape);

  block_setup_.SetupBlocks(strong_shape, true);
  kernels::DynamicScratchpad scratchpad(ws.stream());

  samples_.resize(nsamples);

  auto stream = ws.stream();

  const auto num_boxes = input._num_elements() / 4;

  if (num_boxes == 0) {
    return;
  }

  for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
    samples_[sample_idx].output = output.mutable_tensor<float>(sample_idx);
    samples_[sample_idx].input = input.tensor<float>(sample_idx);
    samples_[sample_idx].horz = horz_[sample_idx].data[0];
    samples_[sample_idx].vert = vert_[sample_idx].data[0];
  }


  GpuBlockSetup::BlockDesc *blocks_dev;
  BbFlipSampleDesc *samples_dev;
  std::tie(samples_dev, blocks_dev) = scratchpad.ToContiguousGPU(ws.stream(),
    samples_, block_setup_.Blocks());

  dim3 grid = block_setup_.GridDim();
  dim3 block = block_setup_.BlockDim();

  if (ltrb_) {
    BbFlipKernel<true><<<grid, block, 0, stream>>>(samples_dev, blocks_dev);
  } else {
    BbFlipKernel<false><<<grid, block, 0, stream>>>(samples_dev, blocks_dev);
  }
  CUDA_CALL(cudaGetLastError());
}

DALI_REGISTER_OPERATOR(BbFlip, BbFlipGPU, GPU);

}  // namespace dali
