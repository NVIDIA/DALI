// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/span.h"
#include "dali/operators/random/rng_base.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/batch_rng.h"
#include "dali/core/static_switch.h"

namespace dali {

struct SampleDesc {
  void *output;
  const void* input;
  int64_t p_count;
  int64_t p_stride;
  int64_t c_count;
  int64_t c_stride;
};

struct BlockDesc {
  int sample_idx;
  int64_t p_offset;
  int64_t p_count;
};

template <bool IsNoiseGen>
struct RNGBaseFields<GPUBackend, IsNoiseGen> {
  RNGBaseFields<GPUBackend, IsNoiseGen>(int64_t seed, int max_batch_size,
                                        int64_t static_sample_size = -1)
      : block_size_(static_sample_size < 0 ? 256 : std::min<int64_t>(static_sample_size, 256)),
        max_blocks_(static_sample_size < 0 ?
                        1024 :
                        std::min<int64_t>(
                            max_batch_size * div_ceil(static_sample_size, block_size_), 1024)),
        randomizer_(seed, block_size_ * max_blocks_) {
    sample_descs_cpu_.resize(max_batch_size);
    sample_descs_gpu_.resize(max_batch_size);
    block_descs_cpu_.resize(max_blocks_);
    block_descs_gpu_.resize(max_blocks_);
  }

  void ReserveDistsData(size_t nbytes) {
    dists_cpu_.reserve(nbytes);
    dists_gpu_.reserve(nbytes);
  }

  const int block_size_;
  const int max_blocks_;
  curand_states randomizer_;

  std::vector<SampleDesc> sample_descs_cpu_;
  DeviceBuffer<SampleDesc> sample_descs_gpu_;

  std::vector<BlockDesc> block_descs_cpu_;
  DeviceBuffer<BlockDesc> block_descs_gpu_;

  std::vector<uint8_t> dists_cpu_;
  DeviceBuffer<uint8_t> dists_gpu_;
};

template <typename Integer>
int DistributeBlocksPerSample(span<int> blocks_per_sample,
                              span<const Integer> sample_sizes,
                              int block_size, int max_blocks) {
  int sum = 0;
  int nsamples = sample_sizes.size();
  assert(nsamples == blocks_per_sample.size());
  for (int i = 0; i < nsamples; ++i) {
    blocks_per_sample[i] = div_ceil(sample_sizes[i], block_size);
    sum += blocks_per_sample[i];
  }
  if (sum <= max_blocks) {
    return sum;
  }
  // If numbers of blocks exceeded max_blocks, we need to scale them down
  int to_distribute = max_blocks - nsamples;  // reserve a block for each sample
  for (int i = 0; i < nsamples; ++i) {
    int before = blocks_per_sample[i];
    int scaled = before * to_distribute / sum;
    // Blocks that are already counted and distributed are subtracted
    // from `sum` and `to_distribute` to make sure that no block is lost
    // due to integer division rounding and at the end all `max_blocks` blocks are distributed.
    to_distribute -= scaled;
    sum -= before;
    blocks_per_sample[i] = scaled + 1;  // each sample gets at least one block
  }
  assert(to_distribute == 0);
  return max_blocks;
}

template <int ndim>
int64_t SetupBlockDescs(BlockDesc *blocks, int64_t block_sz, int64_t max_nblocks,
                        const TensorListShape<ndim> &shape, int channel_dim = -1) {
  int nsamples = shape.num_samples();
  SmallVector<int64_t, 256> sample_sizes;
  sample_sizes.resize(nsamples);
  for (int s = 0; s < nsamples; s++) {
    int64_t npixels = shape.tensor_size(s);
    if (channel_dim >= 0)
      npixels /= shape.tensor_shape_span(s)[channel_dim];
    sample_sizes[s] = npixels;
  }
  SmallVector<int, 256> blocks_per_sample;
  blocks_per_sample.resize(nsamples);
  int64_t blocks_num = DistributeBlocksPerSample(
      make_span(blocks_per_sample), make_cspan(sample_sizes), block_sz, max_nblocks);
  int64_t block = 0;
  for (int s = 0; s < nsamples; s++) {
    auto sample_size = sample_sizes[s];
    if (sample_size == 0)
      continue;
    auto work_per_block = div_ceil(sample_size, blocks_per_sample[s]);
    int64_t offset = 0;
    for (int b = 0; b < blocks_per_sample[s]; ++b, ++block) {
      blocks[block].sample_idx = s;
      blocks[block].p_offset = offset;
      blocks[block].p_count = std::min(work_per_block, sample_size - offset);
      offset += blocks[block].p_count;
    }
  }
  return blocks_num;
}

template <typename T>
void SetupSampleDescs(SampleDesc *samples,
                      TensorListView<StorageGPU, T> &output,
                      TensorListView<StorageGPU, const T> &input,
                      int channel_dim = -1) {
  int nsamples = output.num_samples();
  for (int s = 0; s < nsamples; s++) {
    T *sample_out = static_cast<T*>(output[s].data);
    const T *sample_in = input.empty() ? nullptr : static_cast<const T *>(input[s].data);
    samples[s].output = sample_out;
    samples[s].input = sample_in;
    auto sh = output.shape.tensor_shape_span(s);
    int64_t sample_sz = volume(sh);
    if (channel_dim >= 0) {
      int nchannels = sh[channel_dim];
      samples[s].p_count = sample_sz / nchannels;
      samples[s].p_stride = channel_dim == 0 ? 1 : nchannels;
      samples[s].c_count = nchannels;
      samples[s].c_stride = volume(sh.begin() + channel_dim + 1, sh.end());
    } else {
      samples[s].p_count = sample_sz;
      samples[s].p_stride = 1;
      samples[s].c_count = 1;
      samples[s].c_stride = 1;
    }
  }
}

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_RNG_BASE_GPU_H_
