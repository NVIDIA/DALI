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
#include "dali/kernels/alloc.h"
#include "dali/operators/random/rng_base.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/batch_rng.h"
#include "dali/core/static_switch.h"

namespace dali {

struct BlockDesc {
  int sample_idx;
  void* start;
  size_t size;
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
int64_t SetupBlockDescs(BlockDesc* blocks, int64_t block_sz, int64_t max_nblocks,
                        TensorListView<StorageGPU, T> &output, cudaStream_t stream) {
  std::vector<int> blocks_per_sample;
  int64_t blocks_num;
  auto &shape = output.shape;
  std::tie(blocks_per_sample, blocks_num) = DistributeBlocksPerSample(shape, block_sz, max_nblocks);
  int64_t block = 0;
  for (int s = 0; s < shape.size(); s++) {
    T *sample_data = static_cast<T*>(output[s].data);
    auto sample_size = volume(shape[s]);
    auto work_per_block = div_ceil(sample_size, blocks_per_sample[s]);
    int64_t offset = 0;
    for (int b = 0; b < blocks_per_sample[s]; ++b, ++block) {
      blocks[block].sample_idx = s;
      blocks[block].start = sample_data + offset;
      blocks[block].size = std::min(work_per_block, sample_size - offset);
      offset += blocks[block].size;
    }
  }
  return blocks_num;
}

template <typename T>
int64_t SetupBlockDescsSingleValue(BlockDesc* blocks, int64_t max_nblocks,
                                   TensorListView<StorageGPU, T> &output, cudaStream_t stream) {
  int nsamples = output.shape.size();
  assert(nsamples <= max_nblocks);
  auto sh = output.shape;
  for (int s = 0; s < nsamples; s++) {
    assert(volume(sh[s]) == 1);
    blocks[s].sample_idx = s;
    blocks[s].start = output[s].data;
    blocks[s].size = 1;
  }
  return nsamples;
}

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_RNG_BASE_GPU_H_
