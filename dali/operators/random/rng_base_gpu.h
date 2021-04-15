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

#ifndef DALI_OPERATORS_RANDOM_RNG_BASE_GPU_H_
#define DALI_OPERATORS_RANDOM_RNG_BASE_GPU_H_

#include <utility>
#include <vector>
#include "dali/core/convert.h"
#include "dali/core/dev_buffer.h"
#include "dali/kernels/alloc.h"
#include "dali/operators/random/rng_base.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/batch_rng.h"
#include "dali/core/static_switch.h"

namespace dali {


template <bool NeedsInput>
struct BlockDesc;

template <>
struct BlockDesc<false> {
  int sample_idx;
  void* output;
  size_t size;
};

template <>
struct BlockDesc<true> {
  int sample_idx;
  void* output;
  const void* input;
  size_t offset;
  size_t size;
};

template <bool NeedsInput>
struct RNGBaseFields<GPUBackend, NeedsInput> {
  RNGBaseFields<GPUBackend, NeedsInput>(int64_t seed, int max_batch_size,
                                        int64_t static_sample_size = -1)
      : block_size_(static_sample_size < 0 ? 256 : std::min<int64_t>(static_sample_size, 256)),
        max_blocks_(static_sample_size < 0 ?
                        1024 :
                        std::min<int64_t>(
                            max_batch_size * div_ceil(static_sample_size, block_size_), 1024)),
        randomizer_(seed, block_size_ * max_blocks_) {
    block_descs_gpu_ = kernels::memory::alloc_unique<Block>(kernels::AllocType::GPU, max_blocks_);
    block_descs_cpu_ =
        kernels::memory::alloc_unique<Block>(kernels::AllocType::Pinned, max_blocks_);
  }

  void ReserveDistsData(size_t nbytes) {
    dists_cpu_.reserve(nbytes);
    dists_gpu_.reserve(nbytes);
  }

  using Block = BlockDesc<NeedsInput>;

  const int block_size_;
  const int max_blocks_;
  curand_states randomizer_;
  kernels::memory::KernelUniquePtr<Block> block_descs_gpu_;
  kernels::memory::KernelUniquePtr<Block> block_descs_cpu_;

  std::vector<uint8_t> dists_cpu_;
  DeviceBuffer<uint8_t> dists_gpu_;
};

template <int ndim>
std::pair<std::vector<int>, int> DistributeBlocksPerSample(
    const TensorListShape<ndim> &shape, int block_size, int max_blocks) {
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

template <typename T>
int64_t SetupBlockDescs(BlockDesc<false> *blocks, int64_t block_sz, int64_t max_nblocks,
                        TensorListView<StorageGPU, T> &output,
                        TensorListView<StorageGPU, const T> &input) {
  (void) input;
  std::vector<int> blocks_per_sample;
  int64_t blocks_num;
  auto &shape = output.shape;
  std::tie(blocks_per_sample, blocks_num) = DistributeBlocksPerSample(shape, block_sz, max_nblocks);
  int64_t block = 0;
  for (int s = 0; s < shape.size(); s++) {
    T *sample_data = static_cast<T *>(output[s].data);
    auto sample_size = volume(shape[s]);
    if (sample_size == 0)
      continue;
    auto work_per_block = div_ceil(sample_size, blocks_per_sample[s]);
    int64_t offset = 0;
    for (int b = 0; b < blocks_per_sample[s]; ++b, ++block) {
      blocks[block].sample_idx = s;
      blocks[block].output = sample_data + offset;
      blocks[block].size = std::min(work_per_block, sample_size - offset);
      offset += blocks[block].size;
    }
  }
  return blocks_num;
}

template <typename T>
int64_t SetupBlockDescs(BlockDesc<true> *blocks, int64_t block_sz, int64_t max_nblocks,
                        TensorListView<StorageGPU, T> &output,
                        TensorListView<StorageGPU, const T> &input) {
  std::vector<int> blocks_per_sample;
  int64_t blocks_num;
  auto &shape = output.shape;
  assert(output.shape == input.shape);
  std::tie(blocks_per_sample, blocks_num) = DistributeBlocksPerSample(shape, block_sz, max_nblocks);
  int64_t block = 0;
  for (int s = 0; s < shape.size(); s++) {
    T *sample_out = static_cast<T*>(output[s].data);
    const T *sample_in = static_cast<const T*>(input[s].data);
    auto sample_size = volume(shape[s]);
    if (sample_size == 0)
      continue;
    auto work_per_block = div_ceil(sample_size, blocks_per_sample[s]);
    int64_t offset = 0;
    for (int b = 0; b < blocks_per_sample[s]; ++b, ++block) {
      blocks[block].sample_idx = s;
      blocks[block].output = sample_out;
      blocks[block].input = sample_in;
      blocks[block].offset = offset;
      blocks[block].size = std::min(work_per_block, sample_size - offset);
      offset += blocks[block].size;
    }
  }
  return blocks_num;
}


}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_RNG_BASE_GPU_H_
