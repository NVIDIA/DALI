// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/random/normal_distribution_op.cuh"
#include <vector>
#include <utility>
#include "dali/core/convert.h"

namespace dali {

namespace detail {

template <typename Out, typename RandType>
__global__ void NormalDistKernel(NormalDistributionGpu::BlockDesc *block_descs,
                                 RandomizerGPU randomizer) {
  auto desc = block_descs[blockIdx.x];
  auto out = static_cast<Out*>(desc.sample);
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int x = desc.start + threadIdx.x; x < desc.end; x += blockDim.x) {
    auto norm = randomizer.normal<RandType>(tid);
    out[x] = ConvertSat<Out>(norm * desc.std + desc.mean);
  }
}

template <typename Out, typename RandType>
__global__ void NormalDistSingleValue(NormalDistributionGpu::BlockDesc *descs,
                                      int size, RandomizerGPU randomizer) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= size) return;
  auto desc = descs[tid];
  auto norm = randomizer.normal<RandType>(tid);
  *(static_cast<Out*>(desc.sample)) = ConvertSat<Out>(norm * desc.std + desc.mean);
}

std::pair<std::vector<int>, int> DistributeBlocksPerSample(
    const TensorListShape<> &shape, int block_size, int max_blocks) {
  std::vector<int> sizes(shape.size());
  int sum = 0;
  for (int i = 0; i < shape.size(); ++i) {
    sizes[i] = div_ceil(volume(shape[i]), block_size);
    sum += sizes[i];
  }
  if (sum <= max_blocks) {
    return {sizes, sum};
  }
  // If numbers of blocks exceeded max_blocks, we need to scale them down
  int to_distribute = max_blocks - shape.size();  // reserve a block for each sample
  for (int i = 0; i < shape.size(); ++i) {
    int before = sizes[i];
    int scaled = before * to_distribute / sum;
    // Blocks that are already counted and distributed are subtracted
    // from `sum` and `to_distribute` to make sure that no block is lost
    // due to integer division rounding and at the end all `max_blocks` blocks are distributed.
    to_distribute -= scaled;
    sum -= before;
    sizes[i] = scaled + 1;  // each sample gets at least one block
  }
  assert(to_distribute == 0);
  return {sizes, max_blocks};
}

}  // namespace detail

NormalDistributionGpu::NormalDistributionGpu(const OpSpec &spec)
      : NormalDistribution(spec), randomizer_(seed_, block_size_ * max_blocks_) {
  DALI_ENFORCE(max_batch_size_ <= max_blocks_,
               "Batch size must be smaller than " + std::to_string(max_blocks_));
  block_descs_gpu_ = mem::alloc_unique<BlockDesc>(kernels::AllocType::GPU, max_blocks_);
  block_descs_cpu_ = mem::alloc_unique<BlockDesc>(kernels::AllocType::Pinned, max_blocks_);
}

NormalDistributionGpu::~NormalDistributionGpu() {
  randomizer_.Cleanup();
}

int NormalDistributionGpu::SetupBlockDescs(TensorList<GPUBackend> &output, cudaStream_t stream) {
  std::vector<int> blocks_per_sample;
  int blocks_num;
  auto shape = output.shape();
  std::tie(blocks_per_sample, blocks_num) =
    detail::DistributeBlocksPerSample(shape, block_size_, max_blocks_);
  BlockDesc *blocks = block_descs_cpu_.get();
  int block = 0;
  for (int s = 0; s < shape.size(); ++s) {
    void *sample = output.raw_mutable_tensor(s);
    auto sample_size = volume(shape[s]);
    auto work_per_block = div_ceil(sample_size, blocks_per_sample[s]);
    int64_t start = 0;
    for (int b = 0; b < blocks_per_sample[s]; ++b, ++block) {
      blocks[block].sample = sample;
      blocks[block].mean = mean_[s];
      blocks[block].std = stddev_[s];
      blocks[block].start = start;
      start += work_per_block;
      blocks[block].end = std::min(start, sample_size);
    }
  }
  CUDA_CALL(cudaMemcpyAsync(block_descs_gpu_.get(), block_descs_cpu_.get(),
                            sizeof(BlockDesc) * blocks_num, cudaMemcpyHostToDevice, stream));
  return blocks_num;
}

int NormalDistributionGpu::SetupSingleValueDescs(TensorList<GPUBackend> &output,
                                                  cudaStream_t stream) {
  assert(output.GetElementsNumber() == output.ntensor());
  auto elems = output.ntensor();
  BlockDesc *blocks = block_descs_cpu_.get();
  for (size_t i = 0; i < elems; ++i) {
    blocks[i].sample = output.raw_mutable_tensor(i);
    blocks[i].mean = mean_[i];
    blocks[i].std = stddev_[i];
  }
  CUDA_CALL(cudaMemcpyAsync(block_descs_gpu_.get(), block_descs_cpu_.get(),
                            sizeof(BlockDesc) * elems, cudaMemcpyHostToDevice, stream));
  auto blocks_num = div_ceil(elems, block_size_);
  return blocks_num;
}

int NormalDistributionGpu::SetupDescs(TensorList<GPUBackend> &output, cudaStream_t stream) {
  if (single_value_in_output_) {
    return SetupSingleValueDescs(output, stream);
  } else {
    return SetupBlockDescs(output, stream);
  }
}

void NormalDistributionGpu::LaunchKernel(int blocks_num, int64_t elements, cudaStream_t stream) {
  if (single_value_in_output_) {
    TYPE_SWITCH(dtype_, type2id, DType, DALI_NORMDIST_TYPES, (
      if (sizeof(DType) > 4) {
        detail::NormalDistSingleValue<DType, double>
            <<<blocks_num, block_size_, 0, stream>>>(block_descs_gpu_.get(), elements, randomizer_);
      } else {
        detail::NormalDistSingleValue<DType, float>
            <<<blocks_num, block_size_, 0, stream>>>(block_descs_gpu_.get(), elements, randomizer_);
      }
    ), DALI_FAIL(make_string("Unsupported output type: ", dtype_)));  // NOLINT
  } else {
    TYPE_SWITCH(dtype_, type2id, DType, DALI_NORMDIST_TYPES, (
      if (sizeof(DType) > 4) {
        detail::NormalDistKernel<DType, double>
            <<<blocks_num, block_size_, 0, stream>>>(block_descs_gpu_.get(), randomizer_);
      } else {
        detail::NormalDistKernel<DType, float>
            <<<blocks_num, block_size_, 0, stream>>>(block_descs_gpu_.get(), randomizer_);
      }
    ), DALI_FAIL(make_string("Unsupported output type: ", dtype_)));  // NOLINT
  }
}

void NormalDistributionGpu::RunImpl(workspace_t<GPUBackend> &ws) {
  auto &output = ws.OutputRef<GPUBackend>(0);
  auto stream = ws.stream();
  int blocks_num = SetupDescs(output, stream);
  LaunchKernel(blocks_num, output.GetElementsNumber(), stream);
}

DALI_REGISTER_OPERATOR(NormalDistribution, NormalDistributionGpu, GPU);

}  // namespace dali
