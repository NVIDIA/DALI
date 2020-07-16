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
#include "dali/core/static_switch.h"

#define NORM_TYPES \
  (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float16, float, double)

namespace dali {

namespace detail {

template <typename Out, typename RandType>
__global__ void NormalDistKernel(NormalDistributionGpu::BlockDesc *block_descs,
                                 Randomizer<GPUBackend> randomizer) {
  auto desc = block_descs[blockIdx.x];
  auto out = static_cast<Out*>(desc.sample);
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int x = desc.start + threadIdx.x; x < desc.end; x += blockDim.x) {
    auto norm = randomizer.normal(tid);
    out[x] = ConvertSat<Out>(norm * desc.std + desc.mean);
  }
}

}  // namespace detail

int NormalDistributionGpu::SetupBlockDesc(TensorList<GPUBackend> &output, cudaStream_t stream) {
  std::vector<int> blocks_per_sample;
  int blocks_num;
  auto shape = output.shape();
  std::tie(blocks_per_sample, blocks_num) = DistributeBlocks(shape);
  std::vector<BlockDesc> blocks(blocks_num);
  int block = 0;
  for (int s = 0; s < shape.size(); ++s) {
    void *sample = output.raw_mutable_tensor(s);
    auto sample_size = volume(shape.tensor_shape_span(s));
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
  CUDA_CALL(cudaMemcpyAsync(block_descs_.get(), blocks.data(), sizeof(BlockDesc) * blocks_num,
                            cudaMemcpyHostToDevice));
  return blocks_num;
}

std::pair<std::vector<int>, int> NormalDistributionGpu::DistributeBlocks(
    const TensorListShape<> &shape) {
  std::vector<int> sizes(shape.size());
  int sum = 0;
  for (int i = 0; i < shape.size(); ++i) {
    sizes[i] = div_ceil(volume(shape.tensor_shape_span(i)), block_size_);
    sum += sizes[i];
  }
  if (sum <= max_blocks_) {
    return {sizes, sum};
  }
  // If numbers of blocks exceeded max_blocks_, we need to scale them down
  int to_distribute = max_blocks_ - shape.size();
  for (int i = 0; i < shape.size(); ++i) {
    auto before = sizes[i];
    auto scaled = before * to_distribute / sum;
    to_distribute -= scaled;
    sum -= before;
    sizes[i] = scaled + 1;  // each sample gets at least one block
  }
  return {sizes, max_blocks_};
}

bool NormalDistributionGpu::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const workspace_t<GPUBackend> &ws) {
  AcquireArguments(ws);
  output_desc.resize(detail::kNumOutputs);
  auto shape = GetOutputShape(ws);
  output_desc[0].shape = shape;
  TYPE_SWITCH(dtype_, type2id, DType, NORM_TYPES, (
          output_desc[0].type = TypeTable::GetTypeInfoFromStatic<DType>();
    ), DALI_FAIL(make_string("Unsupported output type: ", dtype_)));  // NOLINT
  return true;
}

void NormalDistributionGpu::RunImpl(workspace_t<GPUBackend> &ws) {
  auto &output = ws.OutputRef<GPUBackend>(0);
  auto stream = ws.stream();
  int blocks_num = SetupBlockDesc(output, stream);
  TYPE_SWITCH(dtype_, type2id, DType, NORM_TYPES, (
    if (sizeof(DType) > 4) {
      detail::NormalDistKernel<DType, double>
          <<<blocks_num, block_size_, 0, stream>>>(block_descs_.get(), randomizer_);
    } else {
      detail::NormalDistKernel<DType, float>
          <<<blocks_num, block_size_, 0, stream>>>(block_descs_.get(), randomizer_);
    }
  ), DALI_FAIL(make_string("Unsupported output type: ", dtype_)));  // NOLINT
}

}  // namespace dali
