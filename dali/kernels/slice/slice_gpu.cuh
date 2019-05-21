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
#include "dali/kernels/common/copy.h"

namespace dali {
namespace kernels {

template <std::size_t Dims>
struct SliceArgsDev {
  DeviceArray<int64_t, Dims> in_strides;
  DeviceArray<int64_t, Dims> out_strides;
  DeviceArray<int64_t, Dims> anchor;
};

template <std::size_t Dims>
struct SliceSampleDesc {
  void *__restrict__ out;
  const void *__restrict__ in;
  DeviceArray<int64_t, Dims> in_strides;
  DeviceArray<int64_t, Dims> out_strides;
  DeviceArray<int64_t, Dims> anchor;
  std::size_t total_size;
};

struct BlockDesc {
  int sampleIdx;
  std::size_t offset;
};

template <typename OutputType, typename InputType, std::size_t Dims>
__global__ void SliceKernelBatched(const SliceSampleDesc<Dims> *samples, const BlockDesc *blocks) {
  int sampleIdx = blocks[blockIdx.x].sampleIdx;
  size_t offset = blocks[blockIdx.x].offset + threadIdx.x;
  auto &sample = samples[sampleIdx];
  auto *out = static_cast<OutputType*>(sample.out);
  auto *in = static_cast<const InputType*>(sample.in);
  size_t total_size = sample.total_size;

  for (; offset < total_size; offset += blockDim.x) {
    unsigned int idx = offset;
    unsigned int out_idx = idx;
    unsigned int in_idx = 0;
    for (std::size_t d = 0; d < Dims; d++) {
      unsigned int i_d = idx / sample.out_strides[d];
      idx = idx % sample.out_strides[d];
      in_idx += (sample.anchor[d] + i_d) * sample.in_strides[d];
    }
    out[out_idx] = clamp<OutputType>(in[in_idx]);
  }
}

template <typename OutputType, typename InputType, std::size_t Dims>
__global__ void SliceKernel(OutputType *__restrict__ output,
                            const InputType *__restrict__ input,
                            SliceArgsDev<Dims> slice_args_dev,
                            unsigned int total_size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size) {
    return;
  }

  unsigned int out_idx = idx;
  unsigned int in_idx = 0;
  for (std::size_t d = 0; d < Dims; d++) {
    unsigned int i_d = idx / slice_args_dev.out_strides[d];
    idx = idx % slice_args_dev.out_strides[d];
    in_idx += (slice_args_dev.anchor[d] + i_d) * slice_args_dev.in_strides[d];
  }
  output[out_idx] = clamp<OutputType>(input[in_idx]);
}

template <typename OutputType, typename InputType, std::size_t Dims>
class DLL_PUBLIC SliceGPU {
 public:
  SliceGPU() = default;

  DLL_PUBLIC KernelRequirements Setup(KernelContext &context,
                                      const InListGPU<InputType, Dims> &in,
                                      const std::vector<SliceArgs<Dims>> &slice_args) {
    KernelRequirements req;

    ScratchpadEstimator se;
    const std::size_t num_samples = in.size();
    se.add<SliceSampleDesc<Dims>>(AllocType::GPU, num_samples);
    se.add<BlockDesc>(AllocType::GPU, num_samples); // TODO(janton): launch more kernels per sample
    req.scratch_sizes = se.sizes;

    req.output_shapes = { GetOutputShapes<Dims>(in.shape, slice_args) };
    return req;
  }

  DLL_PUBLIC void Run(KernelContext &context,
                      OutListGPU<OutputType, Dims> &out,
                      const InListGPU<InputType, Dims> &in,
                      const std::vector<SliceArgs<Dims>> &slice_args) {
    const auto num_samples = in.size();

    std::vector<SliceSampleDesc<Dims>> sample_descs_cpu(num_samples);
    for (int i = 0; i < in.size(); i++) {
      const auto in_shape = in.tensor_shape(i);
      const auto out_shape = out.tensor_shape(i);
      auto &sample_desc = sample_descs_cpu[i];
      sample_desc.in_strides = GetStrides<Dims>(in_shape);
      sample_desc.out_strides = GetStrides<Dims>(out_shape);
      sample_desc.anchor = slice_args[i].anchor;
      sample_desc.in = in.tensor_data(i);
      sample_desc.out = out.tensor_data(i);
      sample_desc.total_size = volume(out_shape);
    }

    SliceSampleDesc<Dims> *sample_descs = context.scratchpad->Allocate<SliceSampleDesc<Dims>>(
      AllocType::GPU, num_samples);
    cudaMemcpyAsync(sample_descs, sample_descs_cpu.data(),
                    num_samples * sizeof(SliceSampleDesc<Dims>),
                    cudaMemcpyHostToDevice,
                    context.gpu.stream);

    std::vector<BlockDesc> block_descs_cpu(num_samples);
    for (int i = 0; i < num_samples; i++) {
        auto &block_desc = block_descs_cpu[i];
        block_desc.sampleIdx = i;
        block_desc.offset = 0;
    }

    BlockDesc *block_descs = context.scratchpad->Allocate<BlockDesc>(
      AllocType::GPU, num_samples);

    cudaMemcpyAsync(block_descs, block_descs_cpu.data(),
                    num_samples * sizeof(BlockDesc),
                    cudaMemcpyHostToDevice,
                    context.gpu.stream);

    const unsigned int grid = block_descs_cpu.size(); // TODO(janton) more blocks!
    const unsigned int block = 1024;
    SliceKernelBatched<OutputType, InputType, Dims>
      <<<grid, block, 0, context.gpu.stream>>>(sample_descs, block_descs);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_GPU_H_
