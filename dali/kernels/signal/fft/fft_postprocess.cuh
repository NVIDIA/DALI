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

#ifndef DALI_KERNELS_SIGNAL_FFT_FFT_POSTPROCESS_CUH_
#define DALI_KERNELS_SIGNAL_FFT_FFT_POSTPROCESS_CUH_

#include <cuda_runtime.h>
#include "dali/core/geom/vec.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {
namespace signal {

namespace fft_postprocess {

template <typename Out, typename In>
struct SampleDesc {
  Out *out;
  const In *in;
  ptrdiff_t out_stride;
  ptrdiff_t in_stride;
};

struct BlockDesc {
  int sample_idx;
  ivec2 start, end;
};

constexpr int kBlock = 32;

struct norm2 {
  DALI_HOST_DEV
  auto operator()(float2 c) const {
    return c.x * c.x + c.y * c.y;
  }
};

struct norm2root {
  DALI_HOST_DEV
  auto operator()(float2 c) const {
#ifdef __CUDA_ARCH__
    return sqrtf(c.x * c.x + c.y * c.y);
#else
    return std::sqrt(c.x * c.x + c.y * c.y);
#endif
  }
};

template <typename Out, typename In, typename Convert = identity>
__global__ void TransposeBatch(
    const SampleDesc<Out, In> *samples,
    const BlockDesc *blocks,
    Convert convert = {}) {
  BlockDesc block = blocks[blockIdx.x];
  SampleDesc<Out, In> sample = samples[block.sample_idx];

  // Use double buffering to cut the use of __syncthreads by half.
  // Without the double buffering, we'd need to wait for tmp to be consumed
  // before being able to populate it again.
  __shared__ Out tmp[2][kBlock][kBlock + 1];

  ivec2 blk, in_pos, out_pos;
  int page = 0;

  for (blk.y = block.start.y; blk.y < block.end.y; blk.y += kBlock) {
    in_pos.y = blk.y + threadIdx.y;
    out_pos.x = blk.y + threadIdx.x;
    for (blk.x = block.start.x; blk.x < block.end.x; blk.x += kBlock) {
      in_pos.x = blk.x + threadIdx.x;
      out_pos.y = blk.x + threadIdx.y;

      if (all_coords(in_pos < block.end)) {
        Out v = sample.in[in_pos.y * sample.in_stride + in_pos.x];
        tmp[page][threadIdx.y][threadIdx.x] = convert(v);
      }

      __syncthreads();
      if (out_pos.y < block.end.x && out_pos.x < block.end.y)
        sample.out[out_pos.y * sample.out_stride + out_pos.x] = tmp[page][threadIdx.x][threadIdx.y];

      page = 1-page;

    }
  }
}

/**
 * A specialized kernel that tranposes a frame-major spectrogrum to frequency-major one,
 * with an option to calculate complex magnitude or squared magnitude.
 *
 * Constraints: all input samples must have the same
 */
template <typename Out, typename In = Out, typename Convert = identity>
struct SpectrumToFreqMajor {
  using SampleDesc = fft_postprocess::SampleDesc<Out, In>;

  KernelRequirements Setup(KernelContext &ctx, const InListGPU<In, 2> &in, int nfft = -1) {
    KernelRequirements req;
    ScratchpadEstimator se;
    req.output_shapes.resize(1);
    auto &out_shape = req.output_shapes[0];

    int N = in.num_samples();
    int nblocks = 0;
    DALI_ENFORCE(nfft <= in.shape[0][1],
      "`nfft` parameter cannot exceed actual data size");

    if (nfft < 0)
      nfft = in.shape[0][1];
    this->nfft_ = nfft;

    int64_t total_windows = 0;
    out_shape.resize(N, 2);
    for (int i = 0; i < N; i++) {
      TensorShape<2> sample_shape = in.shape[i];
      DALI_ENFORCE(sample_shape[1] == nfft, "All inputs must have the same number of FFT bins");
      out_shape.set_tensor_shape(i, { sample_shape[1], sample_shape[0] });

      total_windows += sample_shape[0];
    }

    // if the data is big, use larger blocks
    block_size_ = kBlock * (1 + total_windows / (100000 * kBlock));

    for (int i = 0; i < N; i++) {
      int nwindows = in.shape[i][0];
      nblocks += div_ceil(nwindows, block_size_);
    }
    nblocks_ = nblocks;
    se.add<SampleDesc>(AllocType::GPU, N);
    se.add<BlockDesc>(AllocType::GPU, nblocks);
    se.add<SampleDesc>(AllocType::Host, N);
    se.add<BlockDesc>(AllocType::Host, nblocks);
    req.scratch_sizes = se.sizes;
    return req;
  }

  void Run(KernelContext &ctx, const OutListGPU<Out, 2> &out, const InListGPU<In, 2> &in) {
    int N = in.num_samples();

    SampleDesc *cpu_samples = ctx.scratchpad->Allocate<SampleDesc>(AllocType::Host, N);
    BlockDesc *cpu_blocks = ctx.scratchpad->Allocate<BlockDesc>(AllocType::Host, nblocks_);

    int b = 0;
    for (int i = 0; i < N; i++) {
      TensorShape<2> sample_shape = in.shape[i];
      assert(sample_shape[1] == nfft_);
      int nwindows = sample_shape[0];
      cpu_samples[i] = {
        out.data[i],
        in.data[i],
        nwindows,  // output stride - a row contains all windows
        nfft_      // input stride  - a row contains all frequencies
      };
      for (int start = 0; start < nwindows; start += block_size_) {
        int end = std::min(start + block_size_, nwindows);
        assert(b < nblocks_);
        cpu_blocks[b++] = { i, ivec2(0, start), ivec2(nfft_, end) };
      }
    }
    assert(b == nblocks_);

    SampleDesc *gpu_samples;
    BlockDesc *gpu_blocks;

    std::tie(gpu_samples, gpu_blocks) = ctx.scratchpad->ToContiguousGPU(
        ctx.gpu.stream, make_span(cpu_samples, N), make_span(cpu_blocks, nblocks_));

    dim3 blockDim(kBlock, kBlock, 1);
    TransposeBatch<<<nblocks_, blockDim, 0, ctx.gpu.stream>>>(gpu_samples, gpu_blocks);

  }

 private:
  int nblocks_ = 0;
  int block_size_ = kBlock;
  int nfft_ = 0;
};

}  // fft_postprocess

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_FFT_FFT_TRANSPOSE_CUH_
