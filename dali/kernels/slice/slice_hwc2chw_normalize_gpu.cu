// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <tuple>
#include <type_traits>
#include "dali/core/backend_tags.h"
#include "dali/core/convert.h"
#include "dali/core/error_handling.h"
#include "dali/core/fast_div.h"
#include "dali/core/float16.h"
#include "dali/core/permute.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/slice/slice_hwc2chw_normalize_gpu.h"

namespace dali {
namespace kernels {

namespace slice_flip_normalize {

template <typename Out, typename In>
struct Hwc2ChwSampleDesc {
  Out *__restrict__ out;
  const In *__restrict__ in;
  const float *__restrict__ norm_add;
  const float *__restrict__ norm_mul;
  const Out *__restrict__ fill_values;

  int64_t sample_size;
  uint32_t first_block;

  // Dimensions of the output
  int H, W, C;
  // Dimensions of the input (relevant to input stride)
  int input_W, input_C;

  bool flip_x;
};

// TODO(klecki): Generalize the utility for binsearch indexing of thread blocks with Cast kernel.
inline __device__ uint32_t FindSampleIdx(const uint32_t *first_blocks, uint32_t num_samples) {
  uint32_t i = 0;
  for (uint32_t jump = (1 << (32 - __clz(num_samples) - 1)); jump; jump >>= 1) {
    if (i + jump < num_samples && first_blocks[i + jump] <= blockIdx.x)
      i += jump;
  }
  return i;
}

/** @defgroup Hwc2Chw The Slice Hwc2Chw Normalize Mirror-x Pad-channel kernel
 *
 * Kernel that reads a HWC u8 image and outputs a CHW normalized float image, that can be cropped
 * in Y, X coordinates, mirrored in X coordinate, and the channels can be padded.
 *
 * Overview of the kernel:
 * The image is processed in flattened coordinates. The Y, X stays the same between the interleaved
 * input layout and planar output layout. Assuming 3-channel input, we can look at the input as
 * a sequential stream of values, where we distribute them (sequentially) into 3 output planes.
 * Use a thread block size, that is divisible both by channel number (for the output loop),
 * and 4 (for input loop).
 * The processing steps:
 * 1. [Input loop] Load the linear chunk of input into shared memory, utilizing 4-byte aligned loads
 *    and cast it to float.
 *   a. Unaligned prologue loop - reads the first chunk till we get to address that is aligned with
 *      32 * 4.
 *   b. Main loop - do as many aligned 4byte reads as possible
 *   c. Epilogue loop - read the remaining values that were not possible to read as one 4byte read.
 * 2. Synchronize
 * 3. [Output loop] Each thread corresponds to a (Y, X) sequential offset into a plane, computes
 *    the values for all the channels and writes them.
 *   a. Optionally, mirroring is performed by inverting the X-coordinate in the output offset.
 *   b. Padding the output channels is performed by filling additional planes with fill values.
 *  @{
 */

/**
 * @brief Hwc2Chw Normalize Mirror-x Pad-channel kernel
 * This kernel does not support cropping the x coordinate, so the reads are fully linear.
 */
template <typename Out, typename In, bool enable_mirror, bool enable_pad, int kBlockSize,
          int kStaticChannels>
__global__ void Hwc2ChwNormalize(const Hwc2ChwSampleDesc<Out, In> *samples, uint32_t *first_blocks,
                                 uint32_t num_samples) {
  // TODO(klecki): generalize for wider input types
  static_assert(std::is_same<In, uint8_t>::value, "Only uint8_t supported as input");

  int sample_idx = FindSampleIdx(first_blocks, num_samples);
  const auto sample = samples[sample_idx];
  int64_t start_x = (blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);
  __shared__ float tile[kBlockSize + 32 * 4];

  // Preload the norm values so they are accessed via registers and not from gmem via pointer.
  float norm_mul[kStaticChannels], norm_add[kStaticChannels];

#pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + start_x);
  auto aligned_in_start = align_up(in_start, 32 * 4);
  auto bytes_skipped = ::min(static_cast<int64_t>(aligned_in_start - in_start), end_x - start_x);

  float *aligned_tile = tile + 32 * 4;
  float *prologue_tile = aligned_tile - bytes_skipped;
  const In *prologue_in = sample.in + start_x;

  const uchar4 *aligned_in_char4 =
      reinterpret_cast<const uchar4 *>(sample.in + start_x + bytes_skipped);

  // prologue
  for (int64_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
    prologue_tile[idx] = prologue_in[idx];
  }
  int64_t left_after_prologue = end_x - start_x - bytes_skipped;

  // aligned load
  for (int64_t idx = threadIdx.x; idx < left_after_prologue >> 2; idx += blockDim.x) {
    uchar4 in = aligned_in_char4[idx];
    aligned_tile[idx * 4 + 0] = in.x;
    aligned_tile[idx * 4 + 1] = in.y;
    aligned_tile[idx * 4 + 2] = in.z;
    aligned_tile[idx * 4 + 3] = in.w;
  }
  int64_t processed_in_main = left_after_prologue & -4;  // equivalent to (x / 4) * 4
  int64_t left_after_main = left_after_prologue - processed_in_main;

  // epilogue
  float *epilogue_tile = aligned_tile + processed_in_main;
  const In *epilogue_in = reinterpret_cast<const In *>(aligned_in_char4 + (processed_in_main >> 2));

  for (int64_t idx = threadIdx.x; idx < left_after_main; idx++) {
    epilogue_tile[idx] = epilogue_in[idx];
  }

  __syncthreads();
  const auto *__restrict__ fill_values = static_cast<const Out *>(sample.fill_values);

  // idx is not divided by the static channels (mostly the start_x)
  for (int64_t idx = threadIdx.x + start_x / kStaticChannels, base_x = threadIdx.x;
       idx < end_x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
    // TODO(klecki): forceinline device function
    int64_t out_offset;
    if constexpr (enable_mirror) {
      if (sample.flip_x) {
        int y = idx / sample.W;
        int x = idx - (int64_t)y * sample.W;
        int target_x = sample.W - 1 - x;
        out_offset = (int64_t)y * sample.W + target_x;
      } else {
        out_offset = idx;
      }
    } else {
      out_offset = idx;
    }

#pragma unroll kStaticChannels
    for (int c = 0; c < kStaticChannels; c++) {
      // the kStaticChannels == input_C
      float fpin = prologue_tile[base_x * sample.input_C + c];
      float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
      sample.out[c * sample.H * sample.W + out_offset] = ConvertSat<Out>(fpout);
    }

    if constexpr (enable_pad) {
      for (int c = kStaticChannels; c < sample.C; c++) {
        sample.out[c * sample.H * sample.W + out_offset] = fill_values[c];
      }
    }
  }
}

/**
 * @brief Slice Hwc2Chw Normalize Mirror-x Pad-channel kernel
 * This kernel supports cropping in x-coordinate.
 * It extends the input loop, by utilizing the (unaligned prologue, aligned main loop, epilogue)
 * pattern in a row-by-row loop.
 * Indexing is based on the output coordinates, specifically we read rows for coordinate X
 * between 0 and output H. (The samples are shifted so they always start from 0 in X).
 */
template <typename Out, typename In, bool enable_mirror, bool enable_pad, int kBlockSize,
          int kStaticChannels>
__global__ void SliceHwc2ChwNormalize(const Hwc2ChwSampleDesc<Out, In> *samples,
                                      uint32_t *first_blocks, uint32_t num_samples) {
  // TODO(klecki): generalize for wider input types
  static_assert(std::is_same<In, uint8_t>::value, "Only uint8_t supported as input");

  int sample_idx = FindSampleIdx(first_blocks, num_samples);

  const auto sample = samples[sample_idx];
  int64_t start_x = (blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);

  __shared__ float tile[kBlockSize + 32 * 4];

  // Preload the norm values so they are accessed via registers and not from gmem via pointer.
  float norm_mul[kStaticChannels], norm_add[kStaticChannels];

#pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

  // Strides use the input number of channels without the padding
  int in_stride = sample.input_W * sample.input_C;
  int out_stride = sample.W * sample.input_C;

  // The rows we start and end with, we are indexed by output coordinates
  int y_start = start_x / out_stride;
  int y_end = end_x / out_stride + 1;

  float *tile_row = tile;

  for (int y = y_start; y < y_end; y++) {
    int xc_start, xc_end;

    // The first row doesn't start with 0 due to tiling, the rest do.
    if (y == y_start) {
      xc_start = start_x - y_start * out_stride;

    } else {
      xc_start = 0;
    }

    // Similarly for the end of row for last row
    if (y == y_end - 1) {
      xc_end = end_x - (y_end - 1) * out_stride;
    } else {
      xc_end = out_stride;
    }

    const In *prologue_in = sample.in + y * in_stride + xc_start;

    auto in_start = reinterpret_cast<std::uintptr_t>(prologue_in);
    // align to 4
    auto aligned_in_start = align_up(in_start, 4);
    auto bytes_skipped =
        ::min(static_cast<int32_t>(aligned_in_start - in_start), xc_end - xc_start);

    float *prologue_tile = tile_row;
    float *aligned_tile = tile_row + bytes_skipped;

    const uchar4 *aligned_in_char4 = reinterpret_cast<const uchar4 *>(prologue_in + bytes_skipped);

    // prologue
    for (int64_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
      prologue_tile[idx] = prologue_in[idx];
    }

    int64_t left_after_prologue = xc_end - xc_start - bytes_skipped;

    // aligned load
    for (int64_t idx = threadIdx.x; idx < left_after_prologue >> 2; idx += blockDim.x) {
      uchar4 in = aligned_in_char4[idx];
      aligned_tile[idx * 4 + 0] = in.x;
      aligned_tile[idx * 4 + 1] = in.y;
      aligned_tile[idx * 4 + 2] = in.z;
      aligned_tile[idx * 4 + 3] = in.w;
    }

    int64_t processed_in_main = left_after_prologue & -4;  // equivalent to (x / 4) * 4
    int64_t left_after_main = left_after_prologue - processed_in_main;

    // epilogue
    float *epilogue_tile = aligned_tile + processed_in_main;
    const In *epilogue_in =
        reinterpret_cast<const In *>(aligned_in_char4 + (processed_in_main >> 2));

    for (int64_t idx = threadIdx.x; idx < left_after_main; idx++) {
      epilogue_tile[idx] = epilogue_in[idx];
    }
    tile_row += (xc_end - xc_start);
  }

  __syncthreads();
  const auto *__restrict__ fill_values = static_cast<const Out *>(sample.fill_values);

  for (int64_t idx = threadIdx.x + start_x / kStaticChannels, base_x = threadIdx.x;
       idx < end_x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
    int64_t out_offset;
    if constexpr (enable_mirror) {
      if (sample.flip_x) {
        int y = idx / sample.W;
        int x = idx - (int64_t)y * sample.W;
        int target_x = sample.W - 1 - x;
        out_offset = (int64_t)y * sample.W + target_x;
      } else {
        out_offset = idx;
      }
    } else {
      out_offset = idx;
    }

#pragma unroll kStaticChannels
    for (int c = 0; c < kStaticChannels; c++) {
      // the kStaticChannels == input_C
      float fpin = tile[base_x * sample.input_C + c];
      float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
      sample.out[c * sample.H * sample.W + out_offset] = ConvertSat<Out>(fpout);
    }

    if constexpr (enable_pad) {
      for (int c = kStaticChannels; c < sample.C; c++) {
        sample.out[c * sample.H * sample.W + out_offset] = fill_values[c];
      }
    }
  }
}

/** @} */  // end of Hwc2Chw

// __device__ __half fma(__half x, __half y, __half z) {
//   return ::fma(x, y, z);
// }

// __device__ float fma(float x, float y, float z) {
//   return fmaf(x, y, z);
// }


// HWC -> HWC + pad channels variant
template <typename Out, typename In, bool enable_mirror, bool enable_pad, int kBlockSize,
          int kStaticChannels>
__global__ void Hwc2HwcNormalize(const Hwc2ChwSampleDesc<Out, In> *samples, uint32_t *first_blocks,
                                 uint32_t num_samples) {
  int sample_idx = FindSampleIdx(first_blocks, num_samples);
  const auto sample = samples[sample_idx];

  int64_t start_x = (blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);

  Out norm_mul[kStaticChannels], norm_add[kStaticChannels];

  __shared__ Out tile[kBlockSize + 32 * 4];

#pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

  // TODO(klecki): assumes u8, 3 and 4 channels

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + start_x);
  auto aligned_in_start = align_up(in_start, 32 * 4);
  auto bytes_skipped = aligned_in_start - in_start;

  Out *aligned_tile = tile + 32 * 4;
  Out *prologue_tile = aligned_tile - bytes_skipped;
  const In *prologue_in = sample.in + start_x;


  // float *aligned_tile_u32 = reinterpret_cast<uint32_t*>(aligned_tile);
  const uchar4 *aligned_in_char4 =
      reinterpret_cast<const uchar4 *>(sample.in + start_x + bytes_skipped);

  // prologue
  for (int64_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
    prologue_tile[idx] = prologue_in[idx];
  }

  int64_t left_after_prologue = end_x - start_x - bytes_skipped;

  // aligned load
  for (int64_t idx = threadIdx.x; idx < left_after_prologue / 4; idx += blockDim.x) {
    uchar4 in = aligned_in_char4[idx];
    aligned_tile[idx * 4 + 0] = in.x;
    aligned_tile[idx * 4 + 1] = in.y;
    aligned_tile[idx * 4 + 2] = in.z;
    aligned_tile[idx * 4 + 3] = in.w;
  }

  int64_t processed_in_main = (left_after_prologue / 4) * 4;
  int64_t left_after_main = left_after_prologue - processed_in_main;

  // epilogue
  Out *epilogue_tile = aligned_tile + processed_in_main;
  const In *epilogue_in = reinterpret_cast<const In *>(aligned_in_char4 + processed_in_main / 4);

  for (int64_t idx = threadIdx.x; idx < left_after_main; idx++) {
    epilogue_tile[idx] = epilogue_in[idx];
  }

  __syncthreads();
  const auto *__restrict__ fill_values = static_cast<const Out *>(sample.fill_values);

  // Assuming all samples are padded
  if constexpr (enable_pad) {
    // int64_t block_4 = (kBlockSize / sample.input_C) * sample.C;
    // int64_t sample_size_4 = (sample.sample_size / sample.input_C) * sample.C;
    // int64_t start_x_padded = static_cast<int64_t>(blockIdx.x - sample.first_block) * block_4;
    // int64_t end_x_padded = ::min(start_x_padded + block_4, sample_size_4);
    // int64_t processed = start_x / sample.input_C;


    int64_t block_4 = (kBlockSize / 3) * 4;
    int64_t sample_size_4 = (sample.sample_size / 3) * 4;
    int64_t start_x_padded = static_cast<int64_t>(blockIdx.x - sample.first_block) * block_4;
    int64_t end_x_padded = ::min(start_x_padded + block_4, sample_size_4);

    for (int64_t idx = threadIdx.x + start_x_padded, base_x = threadIdx.x; idx < end_x_padded;
         idx += blockDim.x, base_x += blockDim.x) {
      int base_offset = base_x >> 2;
      int c = idx & 3;

      int64_t out_offset;
      if constexpr (enable_mirror) {
        if (sample.flip_x) {
          int y = idx / (sample.W * sample.C);
          int xc = idx - (int64_t)y * sample.W * sample.C;
          int x = xc >> 2;
          int target_x = sample.W - 1 - x;
          out_offset = (int64_t)y * sample.W * sample.C + target_x * 4 + c;
        } else {
          out_offset = idx;
        }
      } else {
        out_offset = idx;
      }

      if (c < kStaticChannels) {
        Out fpin = prologue_tile[base_offset * sample.input_C + c];
        Out fpout = fma(fpin, norm_mul[c], norm_add[c]);
        sample.out[out_offset] = fpout;
      } else {
        sample.out[out_offset] = fill_values[c];
      }
    }
  } else {
    fast_div<uint32_t> channels(kStaticChannels);
    for (int64_t idx = threadIdx.x + start_x, base_x = threadIdx.x; idx < end_x;
         idx += blockDim.x, base_x += blockDim.x) {
      int c = idx % channels;


      int64_t out_offset;
      if constexpr (enable_mirror) {
        if (sample.flip_x) {
          int y = idx / (sample.W * kStaticChannels);
          int xc = idx - (int64_t)y * sample.W * kStaticChannels;
          int x = xc / channels;
          int target_x = sample.W - 1 - x;
          out_offset = (int64_t)y * sample.W * kStaticChannels + target_x * kStaticChannels + c;
        } else {
          out_offset = idx;
        }
      } else {
        out_offset = idx;
      }

      Out fpin = prologue_tile[base_x];
      Out fpout = fma(fpin, norm_mul[c], norm_add[c]);
      sample.out[out_offset] = ConvertSat<Out>(fpout);
    }
  }
}


// HWC -> HWC + pad channels variant
template <typename Out, typename In, bool enable_mirror, bool enable_pad, int kBlockSize,
          int kStaticChannels>
__global__ void SliceHwc2HwcNormalize(const Hwc2ChwSampleDesc<Out, In> *samples,
                                      uint32_t *first_blocks, uint32_t num_samples) {
  int sample_idx = FindSampleIdx(first_blocks, num_samples);

  const auto sample = samples[sample_idx];
  int64_t start_x = (blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);

  __shared__ Out tile[kBlockSize + 32 * 4];

  // Preload the norm values so they are accessed via registers and not from gmem via pointer.
  Out norm_mul[kStaticChannels], norm_add[kStaticChannels];

#pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

  // Strides use the input number of channels without the padding
  int in_stride = sample.input_W * sample.input_C;
  int out_stride = sample.W * sample.input_C;

  // The rows we start and end with, we are indexed by output coordinates
  int y_start = start_x / out_stride;
  int y_end = end_x / out_stride + 1;

  Out *tile_row = tile;

  for (int y = y_start; y < y_end; y++) {
    int xc_start, xc_end;

    // The first row doesn't start with 0 due to tiling, the rest do.
    if (y == y_start) {
      xc_start = start_x - y_start * out_stride;

    } else {
      xc_start = 0;
    }

    // Similarly for the end of row for last row
    if (y == y_end - 1) {
      xc_end = end_x - (y_end - 1) * out_stride;
    } else {
      xc_end = out_stride;
    }

    const In *prologue_in = sample.in + y * in_stride + xc_start;

    auto in_start = reinterpret_cast<std::uintptr_t>(prologue_in);
    // align to 4
    auto aligned_in_start = align_up(in_start, 4);
    auto bytes_skipped =
        ::min(static_cast<int32_t>(aligned_in_start - in_start), xc_end - xc_start);

    Out *prologue_tile = tile_row;
    Out *aligned_tile = tile_row + bytes_skipped;

    const uchar4 *aligned_in_char4 = reinterpret_cast<const uchar4 *>(prologue_in + bytes_skipped);

    // prologue
    for (int64_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
      prologue_tile[idx] = prologue_in[idx];
    }

    int64_t left_after_prologue = xc_end - xc_start - bytes_skipped;

    // aligned load
    for (int64_t idx = threadIdx.x; idx < left_after_prologue / 4; idx += blockDim.x) {
      uchar4 in = aligned_in_char4[idx];
      aligned_tile[idx * 4 + 0] = in.x;
      aligned_tile[idx * 4 + 1] = in.y;
      aligned_tile[idx * 4 + 2] = in.z;
      aligned_tile[idx * 4 + 3] = in.w;
    }

    int64_t processed_in_main = (left_after_prologue / 4) * 4;
    int64_t left_after_main = left_after_prologue - processed_in_main;

    // epilogue
    Out *epilogue_tile = aligned_tile + processed_in_main;
    const In *epilogue_in = reinterpret_cast<const In *>(aligned_in_char4 + processed_in_main / 4);

    for (int64_t idx = threadIdx.x; idx < left_after_main; idx++) {
      epilogue_tile[idx] = epilogue_in[idx];
    }
    tile_row += (xc_end - xc_start);
  }

  __syncthreads();
  const auto *__restrict__ fill_values = static_cast<const Out *>(sample.fill_values);

  // Assuming all samples are padded
  if constexpr (enable_pad) {
    // int64_t block_4 = (kBlockSize / sample.input_C) * sample.C;
    // int64_t sample_size_4 = (sample.sample_size / sample.input_C) * sample.C;
    // int64_t start_x_padded = static_cast<int64_t>(blockIdx.x - sample.first_block) * block_4;
    // int64_t end_x_padded = ::min(start_x_padded + block_4, sample_size_4);
    // int64_t processed = start_x / sample.input_C;


    int64_t block_4 = (kBlockSize / 3) * 4;
    int64_t sample_size_4 = (sample.sample_size / 3) * 4;
    int64_t start_x_padded = static_cast<int64_t>(blockIdx.x - sample.first_block) * block_4;
    int64_t end_x_padded = ::min(start_x_padded + block_4, sample_size_4);

    for (int64_t idx = threadIdx.x + start_x_padded, base_x = threadIdx.x; idx < end_x_padded;
         idx += blockDim.x, base_x += blockDim.x) {
      int base_offset = base_x >> 2;
      int c = idx & 3;

      int64_t out_offset;
      if constexpr (enable_mirror) {
        if (sample.flip_x) {
          int y = idx / (sample.W * sample.C);
          int xc = idx - (int64_t)y * sample.W * sample.C;
          int x = xc >> 2;
          int target_x = sample.W - 1 - x;
          out_offset = (int64_t)y * sample.W * sample.C + target_x * 4 + c;
        } else {
          out_offset = idx;
        }
      } else {
        out_offset = idx;
      }

      if (c < kStaticChannels) {
        Out fpin = tile[base_offset * sample.input_C + c];
        Out fpout = fma(fpin, norm_mul[c], norm_add[c]);
        sample.out[out_offset] = fpout;
      } else {
        sample.out[out_offset] = fill_values[c];
      }
    }
  } else {
    fast_div<uint32_t> channels(kStaticChannels);
    for (int64_t idx = threadIdx.x + start_x, base_x = threadIdx.x; idx < end_x;
         idx += blockDim.x, base_x += blockDim.x) {
      int c = idx % channels;


      int64_t out_offset;
      if constexpr (enable_mirror) {
        if (sample.flip_x) {
          int y = idx / (sample.W * kStaticChannels);
          int xc = idx - (int64_t)y * sample.W * kStaticChannels;
          int x = xc / channels;
          int target_x = sample.W - 1 - x;
          out_offset = (int64_t)y * sample.W * kStaticChannels + target_x * kStaticChannels + c;
        } else {
          out_offset = idx;
        }
      } else {
        out_offset = idx;
      }

      Out fpin = tile[base_x];
      Out fpout = fma(fpin, norm_mul[c], norm_add[c]);
      sample.out[out_offset] = ConvertSat<Out>(fpout);
    }
  }
}

template <typename Out>
KernelRequirements SliceHwc2ChwNormalizeGPU<Out>::Setup(KernelContext &ctx,
                                                        const TensorListShape<ndim> &input_shape,
                                                        span<const SampleArgs> args,
                                                        std::array<int, ndim> perm) {
  (void)ctx;
  int num_samples = input_shape.num_samples();
  DALI_ENFORCE(num_samples == static_cast<int>(args.size()),
               "Invalid number of samples in kernel args");
  out_shape_ = TensorListShape<ndim>(num_samples, ndim);
  collapsed_tiling_shape_ = TensorListShape<1>(num_samples, 1);

  SetupNumChannels(input_shape, args);
  perm_ = perm;

  for (int i = 0; i < num_samples; i++) {
    // N.B. this function produces a HWC shape, that's why we need the permute
    auto out_sample_shape = ShapeFromRoi(args[i].roi, out_nchannels_);
    for (int d = 0; d < spatial_dim; d++) {
      DALI_ENFORCE(out_sample_shape[d] <= input_shape.tensor_shape_span(i)[d],
                   make_string("Only cropping allowed, got a request for padding in dimension `", d,
                               "` of sample ", i, "."));
    }
    out_sample_shape = permute(out_sample_shape, perm_);
    out_shape_.set_tensor_shape(i, out_sample_shape);
    collapsed_tiling_shape_.set_tensor_shape(i, {volume(args[i].roi) * nchannels_});
  }
  KernelRequirements req;
  req.output_shapes = {out_shape_};
  return req;
}

template <typename Out>
std::tuple<float *, float *, Out *> SliceHwc2ChwNormalizeGPU<Out>::SetupParams(
    KernelContext &ctx, span<const SampleArgs> args) {
  int num_samples = args.size();
  float *norm_add_cpu = ctx.scratchpad->AllocatePinned<float>(num_samples * nchannels_);
  float *norm_mul_cpu = ctx.scratchpad->AllocatePinned<float>(num_samples * nchannels_);
  Out *fill_values_cpu = ctx.scratchpad->AllocatePinned<Out>(num_samples * out_nchannels_);
  for (int i = 0; i < num_samples; i++) {
    const auto &sample_arg = args[i];
    auto *norm_add_data = norm_add_cpu + i * nchannels_;
    auto *norm_mul_data = norm_mul_cpu + i * nchannels_;
    int mean_sz = sample_arg.mean.size();
    assert(mean_sz == sample_arg.inv_stddev.size());
    int c = 0;
    for (; c < mean_sz; c++) {
      norm_add_data[c] = -sample_arg.mean[c] * sample_arg.inv_stddev[c];
      norm_mul_data[c] = sample_arg.inv_stddev[c];
    }
    for (; c < nchannels_; c++) {
      norm_add_data[c] = 0.0f;
      norm_mul_data[c] = 1.0f;
    }
    auto *fill_values_data = fill_values_cpu + i * out_nchannels_;
    int fill_values_sz = sample_arg.fill_values.size();
    c = 0;
    for (; c < fill_values_sz; c++)
      fill_values_data[c] = ConvertSat<Out>(sample_arg.fill_values[c]);
    for (; c < out_nchannels_; c++)
      fill_values_data[c] = ConvertSat<Out>(0.0f);
  }

  return ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream,
                                         make_span(norm_add_cpu, num_samples * nchannels_),
                                         make_span(norm_mul_cpu, num_samples * nchannels_),
                                         make_span(fill_values_cpu, num_samples * out_nchannels_));
}

template <typename Out>
auto SliceHwc2ChwNormalizeGPU<Out>::RealignSample(TensorView<StorageGPU, const In, ndim> in_sample,
                                                  Roi<spatial_dim> roi)
    -> std::tuple<TensorView<StorageGPU, const In, ndim>, Roi<spatial_dim>> {
  const auto *data = in_sample.data;
  auto shape = in_sample.shape;
  // skip the cropped rows
  data += roi.lo.y * shape[1] * shape[2];
  shape[0] = roi.extent().y;
  // skip the cropped columns
  data += roi.lo.x * shape[2];


  return {TensorView<StorageGPU, const In, ndim>{data, shape},
          {ivec<spatial_dim>{0}, roi.extent()}};
}

template <typename Out>
void SliceHwc2ChwNormalizeGPU<Out>::SetupNumChannels(const TensorListShape<ndim> &input_shape,
                                                     span<const SampleArgs> args) {
  if (input_shape.num_samples() == 0) {
    return;
  }
  const auto first_shape = input_shape.tensor_shape_span(0);
  nchannels_ = first_shape[channel_dim];
  for (int i = 1; i < input_shape.num_samples(); i++) {
    int ch = input_shape.tensor_shape_span(i)[channel_dim];
    DALI_ENFORCE(nchannels_ == ch,
                 make_string("All samples should have the same number of channels, expected ",
                             nchannels_, " channels, got ", ch, " channels in sample ", i));
  }
  DALI_ENFORCE(
      input_shape.num_samples() == static_cast<int>(args.size()),
      "Number of samples in the arguments should match the number of samples in the shape");

  out_nchannels_ = std::max(nchannels_, static_cast<int>(args[0].fill_values.size()));
  for (int i = 1; i < input_shape.num_samples(); i++) {
    DALI_ENFORCE(args[i].fill_values.size() == args[0].fill_values.size(),
                 "All sample arguments should have the same number of fill values");
  }
}


template <typename Out>
void SliceHwc2ChwNormalizeGPU<Out>::Run(KernelContext &ctx,
                                        const TensorListView<StorageGPU, Out, ndim> &out,
                                        const TensorListView<StorageGPU, const In, ndim> &in,
                                        span<const SampleArgs> args) {
  using SampleDesc = Hwc2ChwSampleDesc<Out, In>;
  int num_samples = in.num_samples();

  SampleDesc *sample_descs_cpu = ctx.scratchpad->AllocatePinned<SampleDesc>(num_samples);
  uint32_t *first_blocks_cpu = ctx.scratchpad->AllocatePinned<uint32_t>(num_samples);
  auto [norm_add_gpu, norm_mul_gpu, fill_values_gpu] = SetupParams(ctx, args);
  bool need_pad = out_nchannels_ != nchannels_;
  bool need_crop_x = false;
  bool need_flip_x = false;

  uint32_t offset_blk = 0;
  int nonempty_samples = 0;

  for (int sample_id = 0; sample_id < num_samples; sample_id++) {
    auto [in_sample, in_roi] = RealignSample(in[sample_id], args[sample_id].roi);
    // we adjusted the in_roi to start from 0, so roi.extent() == roi.hi
    if (in_sample.shape[1] != in_roi.hi.x) {
      need_crop_x = true;
    }
    int64_t sample_size = collapsed_tiling_shape_[sample_id][0];

    if (sample_size == 0) {
      continue;
    }

    auto &sample_desc = sample_descs_cpu[nonempty_samples];
    auto &first_block = first_blocks_cpu[nonempty_samples++];
    sample_desc.in = in_sample.data;
    sample_desc.out = out.tensor_data(sample_id);

    first_block = offset_blk;
    sample_desc.first_block = offset_blk;
    sample_desc.sample_size = sample_size;
    offset_blk += div_ceil(sample_size, kBlockSizeMul * kBlockWidth);

    // The output shape here is after the permutation
    if (perm_[0] == 2) {  // CHW
      sample_desc.H = out.tensor_shape(sample_id)[1];
      sample_desc.W = out.tensor_shape(sample_id)[2];
      sample_desc.C = out.tensor_shape(sample_id)[0];  // out_nchannels_
    } else {
      sample_desc.H = out.tensor_shape(sample_id)[0];
      sample_desc.W = out.tensor_shape(sample_id)[1];
      sample_desc.C = out.tensor_shape(sample_id)[2];  // out_nchannels_
    }
    sample_desc.input_W = in_sample.shape[1];
    sample_desc.input_C = in_sample.shape[2];  // nchannels_

    sample_desc.norm_add = norm_add_gpu + sample_id * nchannels_;
    sample_desc.norm_mul = norm_mul_gpu + sample_id * nchannels_;
    sample_desc.fill_values = fill_values_gpu + sample_id * out_nchannels_;
    sample_desc.flip_x = args[sample_id].flip_x;
    if (args[sample_id].flip_x) {
      need_flip_x = true;
    }
  }

  if (nonempty_samples == 0)
    return;

  if (perm_[0] == 2) {  // CHW
    auto [sample_descs_gpu, first_blocks_gpu] = ctx.scratchpad->ToContiguousGPU(
        ctx.gpu.stream, make_span(sample_descs_cpu, nonempty_samples),
        make_span(first_blocks_cpu, nonempty_samples));

    auto dispatch = [samples = sample_descs_gpu, blocks = first_blocks_gpu, &ctx, need_crop_x,
                     offset_blk, nonempty_samples](auto pad_v, auto flip_x_v) {
      if (need_crop_x) {
        SliceHwc2ChwNormalize<Out, In, flip_x_v.value, pad_v.value, kBlockSizeMul * kBlockWidth,
                              kStaticChannels><<<offset_blk, kThreadBlockSize, 0, ctx.gpu.stream>>>(
            samples, blocks, nonempty_samples);
      } else {
        Hwc2ChwNormalize<Out, In, flip_x_v.value, pad_v.value, kBlockSizeMul * kBlockWidth,
                         kStaticChannels><<<offset_blk, kThreadBlockSize, 0, ctx.gpu.stream>>>(
            samples, blocks, nonempty_samples);
      }
    };

    auto dispatch_flip = [&](auto pad_v, bool flip_x) {
      if (flip_x) {
        dispatch(pad_v, std::true_type{});
      } else {
        dispatch(pad_v, std::false_type{});
      }
    };

    if (need_pad) {
      dispatch_flip(std::true_type{}, need_flip_x);
    } else {
      dispatch_flip(std::false_type{}, need_flip_x);
    }
  } else {
    auto [sample_descs_gpu, first_blocks_gpu] = ctx.scratchpad->ToContiguousGPU(
        ctx.gpu.stream, make_span(sample_descs_cpu, nonempty_samples),
        make_span(first_blocks_cpu, nonempty_samples));

    auto dispatch = [samples = sample_descs_gpu, blocks = first_blocks_gpu, &ctx, need_crop_x,
                     offset_blk, nonempty_samples](auto pad_v, auto flip_x_v) {
      if (need_crop_x) {
        SliceHwc2HwcNormalize<Out, In, flip_x_v.value, pad_v.value, kBlockSizeMul * kBlockWidth,
                              kStaticChannels><<<offset_blk, kThreadBlockSize, 0, ctx.gpu.stream>>>(
            samples, blocks, nonempty_samples);
      } else {
        Hwc2HwcNormalize<Out, In, flip_x_v.value, pad_v.value, kBlockSizeMul * kBlockWidth,
                         kStaticChannels><<<offset_blk, kThreadBlockSize, 0, ctx.gpu.stream>>>(
            samples, blocks, nonempty_samples);
      }
    };

    auto dispatch_flip = [&](auto pad_v, bool flip_x) {
      if (flip_x) {
        dispatch(pad_v, std::true_type{});
      } else {
        dispatch(pad_v, std::false_type{});
      }
    };

    if (need_pad) {
      dispatch_flip(std::true_type{}, need_flip_x);
    } else {
      dispatch_flip(std::false_type{}, need_flip_x);
    }
  }


  CUDA_CALL(cudaGetLastError());
}


template class DLL_PUBLIC SliceHwc2ChwNormalizeGPU<float>;
template class DLL_PUBLIC SliceHwc2ChwNormalizeGPU<float16>;

}  // namespace slice_flip_normalize

}  // namespace kernels
}  // namespace dali
