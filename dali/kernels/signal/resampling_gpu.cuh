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
#include "dali/core/util.h"
#include "dali/core/convert.h"

#define SHM_NCHANNELS 16

namespace dali {
namespace kernels {
namespace signal {

namespace resampling {

struct SampleDesc {
  void *out;
  const void *in;
  ResamplingWindow window;
  int64_t in_len;  // num samples in input
  int64_t out_begin;  // output region-of-interest start
  int64_t out_end;  // output region-of-interest end
  int nchannels;  // number of channels
  double scale;  // in_sampling_rate / out_sampling_rate
};

/**
 * @brief Gets intermediate floating point representation depending on the input/output types
 */
template <typename Out, typename In>
__device__ float ConvertInput(In in_val) {
  if (std::is_unsigned<Out>::value && std::is_signed<In>::value) {
    return (ConvertSatNorm<float>(in_val) + 1.0f) * 0.5f;
  } else if (std::is_signed<Out>::value && std::is_unsigned<In>::value) {
    return ConvertSatNorm<float>(in_val) * 2.0f - 1.0f;  // treat half-range as 0
  } else {
    return ConvertSatNorm<float>(in_val);  // just normalize
  }
}

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
  auto& window = sample.window;

  extern __shared__ float sh_mem[];
  float *window_coeffs_sh = sh_mem;
  float *tmp = sh_mem + window.lookup_size +
               threadIdx.x * (SHM_NCHANNELS+1);  // used to accummulate per-channel out values
  for (int k = threadIdx.x; k < window.lookup_size; k += blockDim.x) {
    window_coeffs_sh[k] = window.lookup[k];
  }
  __syncthreads();
  window.lookup = window_coeffs_sh;

  Out* out = reinterpret_cast<Out*>(sample.out);
  const In* in = reinterpret_cast<const In*>(sample.in);

  int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t out_block_start = sample.out_begin + static_cast<int64_t>(blockIdx.x) * blockDim.x;

  for (int64_t out_pos = out_block_start + threadIdx.x; out_pos < sample.out_end;
       out_block_start += grid_stride, out_pos += grid_stride) {
    // A floating point distance `in_pos_start` is calculated from an arbitrary relative
    // position, keeping the floats small in order to keep precision. `in_block_f`, used to
    // calculate the reference for distance (in_block_i) needs to be calculated in double precision.
    double in_block_f = out_block_start * scale;
    int64_t in_block_i = ::floor(in_block_f);
    float in_pos_start = in_block_f - in_block_i;
    const In* in_blk_ptr = in + in_block_i * nchannels;
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
        float in_val = ConvertInput<Out, In>(in_blk_ptr[i]);
        float x = i - in_pos;
        float w = window(x);
        out_val = fma(in_val, w, out_val);
      }
      out[out_pos - sample.out_begin] = ConvertSatNorm<Out>(out_val);
    } else {  // multiple channels
      Out *out_ptr = out + (out_pos - sample.out_begin) * nchannels;
      for (int c0 = 0; c0 < nchannels; c0 += SHM_NCHANNELS) {
        int nc = cuda_min(SHM_NCHANNELS, nchannels - c0);
        for (int c = 0; c < nc; c++) {
          tmp[c] = 0;
        }

        for (int i = i0; i < i1; i++) {
          float x = i - in_pos;
          float w = window(x);
          const In *in_ptr = in_blk_ptr + i * nchannels + c0;
          for (int c = 0; c < nc; c++) {
            float in_val = ConvertInput<Out, In>(in_ptr[c]);
            tmp[c] = fma(in_val, w, tmp[c]);
          }
        }

        for (int c = 0; c < nc; c++) {
          out_ptr[c0 + c] = ConvertSatNorm<Out>(tmp[c]);
        }
      }
    }
  }
}

}  // namespace resampling
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_RESAMPLING_GPU_CUH_
