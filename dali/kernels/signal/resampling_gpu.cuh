// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_SIGNAL_RESAMPLING_GPU_CUH_
#define DALI_KERNELS_SIGNAL_RESAMPLING_GPU_CUH_

#include <cuda_runtime.h>
#include "dali/kernels/signal/resampling.h"

namespace dali {
namespace kernels {
namespace signal {

namespace resampling {

struct SampleDesc {
  void *out;
  const void *in;
  ResamplingWindow window;
  int64_t in_len;  // num samples in input
  int64_t out_len;  // num samples in output
  int64_t nchannels;  // number of channels
  double scale;  // in_sampling_rate / out_sampling_rate
};

/**
 * @brief Resamples 1D signal (single or multi-channel), optionally converting to a different data type.
 *
 * @param samples sample descriptors
 */
template <typename Out, typename In, bool SingleChannel = false>
__global__ void ResampleGPUKernel(const SampleDesc *samples) {
  auto sample = samples[blockIdx.y];
  double scale = sample.scale;
  float fscale = scale;
  int nchannels = SingleChannel ? 1 : sample.nchannels;
  auto &window = sample.window;

  Out* out = reinterpret_cast<Out*>(sample.out);
  const In* in = reinterpret_cast<const In*>(sample.in);

  int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t out_block = static_cast<int64_t>(blockIdx.x) * blockDim.x;
  int64_t start_out_pos = out_block + threadIdx.x;

  double in_block_f = out_block * scale;
  int64_t in_block_i = std::floor(in_block_f);
  float in_pos_start = in_block_f - in_block_i;
  const In* in_blk_ptr = in + in_block_i * nchannels;

  for (int64_t out_pos = start_out_pos; out_pos < sample.out_len;
       out_pos += grid_stride, in_pos_start += fscale * grid_stride) {
    float in_pos = in_pos_start + fscale * threadIdx.x;
    auto i_range = window.input_range(in_pos);
    int i0 = i_range.i0;
    int i1 = i_range.i1;
    if (i0 + in_block_i < 0)
      i0 = -in_block_i;
    if (i1 + in_block_i > sample.in_len)
      i1 = sample.in_len - in_block_i;

    float out_val = 0;
    if (SingleChannel) {
      for (int i = i0; i < i1; i++) {
        In in_val = in_blk_ptr[i];
        float x = i - in_pos;
        float w = window(x);
        out_val = fma(in_val, w, out_val);
      }
      out[out_pos] = ConvertSatNorm<Out>(out_val);
    } else {  // multiple channels
      assert(nchannels <= 32);
      float tmp[32];  // more than enough
      for (int c = 0; c < nchannels; c++) {
        tmp[c] = 0;
      }

      for (int i = i0; i < i1; i++) {
        float x = i - in_pos;
        float w = window(x);
        for (int c = 0; c < nchannels; c++) {
          In in_val = in_blk_ptr[i * nchannels + c];
          tmp[c] = fma(in_val, w, tmp[c]);
        }
      }

      for (int c = 0; c < nchannels; c++) {
        out[out_pos * nchannels + c] = ConvertSatNorm<Out>(tmp[c]);
      }
    }
  }
}

}  // namespace resampling
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_RESAMPLING_GPU_CUH_
