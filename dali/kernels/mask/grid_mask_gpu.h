// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_MASK_GRID_MASK_GPU_H_
#define DALI_KERNELS_MASK_GRID_MASK_GPU_H_

#include <cuda_runtime.h>
#include <vector>
#include "dali/kernels/kernel.h"
#include "dali/kernels/common/block_setup.h"

namespace dali {
namespace kernels {

template <typename T>
struct GridMaskSampleDesc {
  T *out;
  const T *in;
  unsigned width, height, channels;
  float ca, sa, sx, sy, ratio;
};

template <typename T, unsigned C>
__global__ void GridMaskKernel(const GridMaskSampleDesc<T> *samples,
                               const BlockDesc<2> *blocks) {
  auto block = blocks[blockIdx.x];
  auto sample = samples[block.sample_idx];

  for (int y = block.start.y + threadIdx.y; y < block.end.y; y += blockDim.y) {
    int x = block.start.x + threadIdx.x;
    int off = (C ? C : sample.channels) * (x + y * sample.width);
    float fxy = -fmaf(y, sample.sa,  sample.sx);
    float fyy =  fmaf(y, sample.ca, -sample.sy);
    for (; x < block.end.x; x += blockDim.x) {
      float fx = fmaf(x, sample.ca, fxy);
      float fy = fmaf(x, sample.sa, fyy);
      if ((fx - ::floor(fx) >= sample.ratio) || (fy - ::floor(fy) >= sample.ratio)) {
        for (unsigned i = 0; i < (C ? C : sample.channels); i++)
          sample.out[off + i] = sample.in[off + i];
      } else {
        for (unsigned i = 0; i < (C ? C : sample.channels); i++)
          sample.out[off + i] = 0;
      }
      off += C * blockDim.x;
    }
  }
}

template <typename Type>
class GridMaskGpu {
  using SampleDesc = GridMaskSampleDesc<Type>;
  using BlockDesc = kernels::BlockDesc<2>;
  std::vector<SampleDesc> sample_descs_;
  BlockSetup<2, 2> block_setup_;

 public:
  KernelRequirements Setup(KernelContext &context, const InListGPU<Type, 3> &in) {
    KernelRequirements req;
    req.output_shapes = {in.shape};
    ScratchpadEstimator se;
    block_setup_.SetupBlocks(in.shape, true);
    se.add<mm::memory_kind::device, SampleDesc>(in.num_samples());
    se.add<mm::memory_kind::device, BlockDesc>(block_setup_.Blocks().size());
    req.scratch_sizes = se.sizes;
    return req;
  }

  void Run(KernelContext &ctx, OutListGPU<Type, 3> &out, const InListGPU<Type, 3> &in,
           const std::vector<int> &tile,
           const std::vector<float> &ratio,
           const std::vector<float> &angle,
           const std::vector<float> &sx,
           const std::vector<float> &sy) {
    int n = in.num_samples();
    int c = in.tensor_shape(0)[2];

    sample_descs_.resize(n);
    for (int i = 0; i < n; i++) {
      const auto &shape = in.tensor_shape(i);
      // if the number of channels is not uniform in the batch, use the
      // non-const version of the kernel.
      if (c != shape[2]) c = 0;

      sample_descs_[i].out = out[i].data;
      sample_descs_[i].in = in[i].data;
      sample_descs_[i].channels = shape[2];
      sample_descs_[i].width = shape[1];
      sample_descs_[i].height = shape[0];

      sample_descs_[i].ca = cos(angle[i]) / tile[i];
      sample_descs_[i].sa = sin(angle[i]) / tile[i];
      sample_descs_[i].sx = sx[i] / tile[i];
      sample_descs_[i].sy = sy[i] / tile[i];
      sample_descs_[i].ratio = ratio[i];
    }

    SampleDesc *samples_gpu;
    BlockDesc *blocks_gpu;
    std::tie(samples_gpu, blocks_gpu) = ctx.scratchpad->ToContiguousGPU(
        ctx.gpu.stream, sample_descs_, block_setup_.Blocks());
    dim3 grid = block_setup_.GridDim();
    dim3 block = block_setup_.BlockDim();
    VALUE_SWITCH(c, C, (1, 2, 3, 4), (
      GridMaskKernel<Type, C><<<grid, block, 0, ctx.gpu.stream>>>(samples_gpu, blocks_gpu);
    ), (  // NOLINT
      GridMaskKernel<Type, 0><<<grid, block, 0, ctx.gpu.stream>>>(samples_gpu, blocks_gpu);
    ));  // NOLINT
    CUDA_CALL(cudaGetLastError());
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_MASK_GRID_MASK_GPU_H_
