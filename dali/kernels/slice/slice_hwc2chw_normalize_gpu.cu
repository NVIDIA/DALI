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
struct Hwc2HwcChwSampleDesc {
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

/** @defgroup Hwc2HwcChwLoad Data loading for slice Hwc2{Hwc,Chw} Normalize Mirror-x Pad-channel
 * kernel Load the data from linear chunk of HWC u8 image into a tile in shared memory. The loading
 * loop consists of three stages:
 * 1. Prologue - read from the start of the tile to the address that is multiple of 4 byte alignment
 * 2. Main loop - read most of the tile via uchar4, utilizing 4-byte read instructions.
 * 3. Epilogue - read the remainder of data that is not covered by the two previous loops.
 *
 * The slicing variant is addressed reads only the values required by the output, proceeding
 * row by row, using the same pattern as above for each row.
 * Samples are adjusted so that rows slices start at 0, and only the end of row is sliced.
 * @{
 */

/**
 * @brief Load the linear tile into linear smem buffer.
 *
 * @tparam kBlockSize Tile size
 * @tparam kStaticChannels Number of input channels
 * @tparam Tile Type of the data kept after loading in the smem tile.
 * @tparam Out Output data type
 * @tparam In Input data type
 * @tparam kLoadAlign - Alignment (in bytes) of the main loop. The access to smem is also aligned
 * to this value, so depending on the prologue length, the data after loading may not start
 * at the tile[0]. The start of actual data is returned.
 * The smem tile must hold at least kBlockSize + kLoadAlign elements.
 * @param tile Shared memory where to load the data.
 * @param sample Sample description
 * @return Tile * - the pointer to the smem where the start of the loaded data is.
 */
template <int kBlockSize, int kStaticChannels, typename Tile, typename Out, typename In,
          int kLoadAlign = 32 * 4>
__device__ __forceinline__ Tile *load_linear_tile(Tile *tile,
                                                  const Hwc2HwcChwSampleDesc<Out, In> sample) {
  static_assert(std::is_same_v<In, uint8_t>, "Only uint8_t types allowed now.");
  static_assert(kStaticChannels == 3, "Only 3 input channels allowed now.");
  static_assert(kLoadAlign % 4 == 0, "The loading alignment should be divisible by 4.");

  int64_t start_x = static_cast<int64_t>(blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + start_x);
  auto aligned_in_start = align_up(in_start, kLoadAlign);

  // In case if end_x - start_x < kLoadAlign, we never get to the aligned main loop
  uint32_t bytes_to_alignment = ::min(aligned_in_start - in_start, end_x - start_x);

  Tile *aligned_tile = tile + kLoadAlign;
  Tile *prologue_tile = aligned_tile - bytes_to_alignment;
  const In *prologue_in = sample.in + start_x;

  const uchar4 *aligned_in_uchar4 =
      reinterpret_cast<const uchar4 *>(sample.in + start_x + bytes_to_alignment);

  // prologue
  for (uint32_t idx = threadIdx.x; idx < bytes_to_alignment; idx += blockDim.x) {
    prologue_tile[idx] = prologue_in[idx];
  }

  // this might be 0, as the prologue may be the full extend of the tile
  uint32_t left_after_prologue = end_x - start_x - bytes_to_alignment;

  // We read 4 values in each iteration
  uint32_t main_loop_length = left_after_prologue >> 2;

  // main loop: aligned load
  for (uint32_t idx = threadIdx.x; idx < main_loop_length; idx += blockDim.x) {
    uchar4 in = aligned_in_uchar4[idx];
    aligned_tile[idx * 4 + 0] = in.x;
    aligned_tile[idx * 4 + 1] = in.y;
    aligned_tile[idx * 4 + 2] = in.z;
    aligned_tile[idx * 4 + 3] = in.w;
  }

  uint32_t processed_in_main = left_after_prologue & ~0x3;;  // equivalent to (x / 4) * 4
  uint32_t left_after_main = left_after_prologue - processed_in_main;

  // epilogue
  Tile *epilogue_tile = aligned_tile + processed_in_main;
  const In *epilogue_in = reinterpret_cast<const In *>(aligned_in_uchar4 + main_loop_length);

  for (uint32_t idx = threadIdx.x; idx < left_after_main; idx++) {
    epilogue_tile[idx] = epilogue_in[idx];
  }

  // Return the start of the tile
  return prologue_tile;
}

/**
 * @brief Load the slices of linear tile into linear smem buffer.
 *
 * The kernel proceeds row-by-row, reading the output width elements/pixels, skipping the remaining
 * input_width - output_width pixels.
 *
 * @tparam kBlockSize Tile size
 * @tparam kStaticChannels Number of input channels
 * @tparam Tile Type of the data kept after loading in the smem tile.
 * @tparam Out Output data type
 * @tparam In Input data type
 * @tparam kLoadAlign - Alignment (in bytes) of the main loop.
 * The smem tile must hold at least kBlockSize + kLoadAlign elements.
 * @param tile Shared memory where to load the data.
 * @param sample Sample description
 * @return Tile * - the pointer to the smem where the start of the loaded data is.
 */
template <int kBlockSize, int kStaticChannels, typename Tile, typename Out, typename In,
          int kLoadAlign = 4>
__device__ __forceinline__ Tile *slice_load_linear_tile(
    Tile *tile, const Hwc2HwcChwSampleDesc<Out, In> sample) {
  static_assert(std::is_same_v<In, uint8_t>, "Only uint8_t types allowed now.");
  static_assert(kStaticChannels == 3, "Only 3 input channels allowed now.");
  static_assert(kLoadAlign % 4 == 0, "The loading alignment should be divisible by 4.");

  int64_t start_x = static_cast<int64_t>(blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);

  // Strides use the input number of channels without the padding
  int in_stride = sample.input_W * sample.input_C;
  // this is intermediate stride, as if we were never padding the data,
  // so it is useful for filling the linear tile, keeping the xy offset
  int tile_stride = sample.W * sample.input_C;

  // The rows we start and end with, we are indexed by output coordinates
  int y_start = start_x / tile_stride;
  int y_end = end_x / tile_stride + 1;

  Tile *tile_row = tile;

  for (int y = y_start; y < y_end; y++) {
    int xc_start, xc_end;

    // The first row doesn't start with 0 due to tiling, the rest do.
    if (y == y_start) {
      xc_start = start_x - y_start * tile_stride;

    } else {
      xc_start = 0;
    }

    // Similarly for the end of row for last row
    if (y == y_end - 1) {
      xc_end = end_x - (y_end - 1) * tile_stride;
    } else {
      xc_end = tile_stride;
    }

    const In *prologue_in = sample.in + y * in_stride + xc_start;

    auto in_start = reinterpret_cast<std::uintptr_t>(prologue_in);
    // align to 4
    auto aligned_in_start = align_up(in_start, kLoadAlign);
    uint32_t bytes_to_alignment =
        ::min(static_cast<int32_t>(aligned_in_start - in_start), xc_end - xc_start);

    Tile *prologue_tile = tile_row;
    Tile *aligned_tile = tile_row + bytes_to_alignment;

    const uchar4 *aligned_in_uchar4 =
        reinterpret_cast<const uchar4 *>(prologue_in + bytes_to_alignment);

    // prologue
    for (uint32_t idx = threadIdx.x; idx < bytes_to_alignment; idx += blockDim.x) {
      prologue_tile[idx] = prologue_in[idx];
    }


    // this might be 0, as the prologue may be the full extend of the tile
    uint32_t left_after_prologue = xc_end - xc_start - bytes_to_alignment;

    // We read 4 values in each iteration
    uint32_t main_loop_length = left_after_prologue >> 2;

    // aligned load
    for (uint32_t idx = threadIdx.x; idx < main_loop_length; idx += blockDim.x) {
      uchar4 in = aligned_in_uchar4[idx];
      aligned_tile[idx * 4 + 0] = in.x;
      aligned_tile[idx * 4 + 1] = in.y;
      aligned_tile[idx * 4 + 2] = in.z;
      aligned_tile[idx * 4 + 3] = in.w;
    }

    uint32_t processed_in_main = left_after_prologue & ~0x3;  // equivalent to (x / 4) * 4
    uint32_t left_after_main = left_after_prologue - processed_in_main;

    // epilogue
    Tile *epilogue_tile = aligned_tile + processed_in_main;
    const In *epilogue_in = reinterpret_cast<const In *>(aligned_in_uchar4 + main_loop_length);

    for (uint32_t idx = threadIdx.x; idx < left_after_main; idx++) {
      epilogue_tile[idx] = epilogue_in[idx];
    }
    tile_row += (xc_end - xc_start);
  }
  return tile;
}

/**
 * @brief Load the slices of linear tile into planar smem buffers.
 *
 * During the loading the values are distributed into separate planes in smem (keeping the same
 * sequential XY coordinates/offsets). Allows for faster access when building padded HWC output.
 * Each smem plane must hold kBlockSize / kStaticChannels elements.
 *
 * @tparam kBlockSize Tile size
 * @tparam kStaticChannels Number of input channels
 * @tparam Tile Type of the data kept after loading in the smem tile.
 * @tparam Out Output data type
 * @tparam In Input data type
 * @tparam kLoadAlign - Alignment (in bytes) of the main loop.
 * @param tile Shared memory where to load the data.
 * @param sample Sample description
 * @return Tile * - the pointer to the smem where the start of the loaded data is.
 */
template <int kBlockSize, int kStaticChannels, typename Tile, typename Out, typename In,
          int kLoadAlign = 4>
__device__ __forceinline__ void load_planar_tile(Tile tile[][kBlockSize / kStaticChannels],
                                                 const Hwc2HwcChwSampleDesc<Out, In> sample) {
  static_assert(std::is_same_v<In, uint8_t>, "Only uint8_t types allowed now.");
  static_assert(kStaticChannels == 3, "Only 3 input channels allowed now.");
  static_assert(kLoadAlign % 4 == 0, "The loading alignment should be divisible by 4.");

  int64_t start_x = static_cast<int64_t>(blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + start_x);
  auto aligned_in_start = align_up(in_start, kLoadAlign);
  uint32_t bytes_to_alignment = ::min(aligned_in_start - in_start, end_x - start_x);

  const In *prologue_in = sample.in + start_x;

  const uchar4 *aligned_in_char4 =
      reinterpret_cast<const uchar4 *>(sample.in + start_x + bytes_to_alignment);

  // The tiles are multiple of 3, so we are always reading from the start of the pixel.

  fast_div<uint32_t> channel(kStaticChannels);
  // prologue
  for (uint32_t idx = threadIdx.x; idx < bytes_to_alignment; idx += blockDim.x) {
    uint32_t xy, c;
    xy = div_mod(c, idx, channel);
    tile[c][xy] = prologue_in[idx];
  }

  // this might be 0, as the prologue may be the full extend of the tile
  uint32_t left_after_prologue = end_x - start_x - bytes_to_alignment;


  // We read 4 values in each iteration
  uint32_t main_loop_length = left_after_prologue >> 2;

  // main loop: aligned load and unpacking
  for (uint32_t idx = threadIdx.x; idx < main_loop_length; idx += blockDim.x) {
    uint32_t flat_idx = idx * 4 + bytes_to_alignment;
    uint32_t xy, c;
    xy = div_mod(c, flat_idx, channel);
    uchar4 in = aligned_in_char4[idx];

    tile[c][xy] = in.x;

    c++;
    if (c == kStaticChannels) {
      c = 0;
      xy++;
    }
    tile[c][xy] = in.y;

    c++;
    if (c == kStaticChannels) {
      c = 0;
      xy++;
    }
    tile[c][xy] = in.z;


    c++;
    if (c == kStaticChannels) {
      c = 0;
      xy++;
    }
    tile[c][xy] = in.w;
  }

  uint32_t processed_in_main = left_after_prologue & ~0x3;  // equivalent to (x / 4) * 4
  uint32_t left_after_main = left_after_prologue - processed_in_main;

  // epilogue
  const In *epilogue_in = reinterpret_cast<const In *>(aligned_in_char4 + main_loop_length);

  for (uint32_t idx = threadIdx.x; idx < left_after_main; idx++) {
    uint32_t flat_idx = processed_in_main + bytes_to_alignment + idx;
    uint32_t xy, c;
    xy = div_mod(c, flat_idx, channel);
    tile[c][xy] = epilogue_in[idx];
  }
}


/** @} */  // end of Hwc2HwcChwLoad


/** @defgroup Hwc2HwcChwStore Data storing for slice Hwc2{Hwc,Chw} Normalize Mirror-x Pad-channel
 * kernel
 * @{
 */

/**
 * @brief Calculate the planar output offset to take optional mirroring into account.
 */
template <bool enable_mirror, typename Out, typename In>
__device__ __forceinline__ int64_t
calculate_offset_chw(int64_t planar_idx, const Hwc2HwcChwSampleDesc<Out, In> sample) {
  if constexpr (enable_mirror) {
    if (sample.flip_x) {
      int y = planar_idx / sample.W;
      int x = planar_idx - (int64_t)y * sample.W;
      int target_x = sample.W - 1 - x;
      return (int64_t)y * sample.W + target_x;
    }
  }
  return planar_idx;
}

template <int kBlockSize, int kStaticChannels, bool enable_mirror, bool enable_pad,
          typename Compute = float, typename Tile, typename Out, typename In>
__device__ __forceinline__ void store_chw(Tile *tile, const Hwc2HwcChwSampleDesc<Out, In> sample) {
  int64_t start_x = static_cast<int64_t>(blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);

  const auto *__restrict__ fill_values = static_cast<const Out *>(sample.fill_values);

  // Preload the norm values so they are accessed via registers and not from gmem via pointer.
  Compute norm_mul[kStaticChannels], norm_add[kStaticChannels];

#pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

  // idx is not divided by the static channels (mostly the start_x)
  for (int64_t idx = threadIdx.x + start_x / kStaticChannels, base_x = threadIdx.x;
       idx < end_x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
    int64_t out_offset = calculate_offset_chw<enable_mirror>(idx, sample);

#pragma unroll kStaticChannels
    for (int c = 0; c < kStaticChannels; c++) {
      // the kStaticChannels == input_C
      Compute fpin = tile[base_x * sample.input_C + c];
      Compute fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
      sample.out[c * sample.H * sample.W + out_offset] = ConvertSat<Out>(fpout);
    }

    if constexpr (enable_pad) {
      for (int c = kStaticChannels; c < sample.C; c++) {
        sample.out[c * sample.H * sample.W + out_offset] = fill_values[c];
      }
    }
  }
}

template <int kOutChannels>
__device__ __forceinline__ int divide_by_channel(int xc) {
  if constexpr (kOutChannels == 3) {
    return xc / kOutChannels;
  }
  return xc >> 2;
}

/**
 * @brief Calculate the flat output offset for interleaved images to take optional mirroring into
 * account.
 */
template <bool enable_mirror, bool enable_pad, typename Out, typename In>
__device__ __forceinline__ int64_t
calculate_offset_hwc(int64_t flat_idx, int c, const Hwc2HwcChwSampleDesc<Out, In> sample) {
  constexpr int kOutChannels = enable_pad ? 4 : 3;
  if constexpr (enable_mirror) {
    if (sample.flip_x) {
      int y = flat_idx / (sample.W * kOutChannels);
      int xc = flat_idx - (int64_t)y * sample.W * kOutChannels;
      int x = divide_by_channel<kOutChannels>(xc);
      int target_x = sample.W - 1 - x;
      return (int64_t)y * sample.W * kOutChannels + target_x * kOutChannels + c;
    }
  }
  return flat_idx;
}

// TODO(klecki): Prepare a generic version that supports the planar layout in smem and evaluate.
template <int kBlockSize, int kStaticChannels, bool enable_mirror, bool enable_pad,
          typename Compute, typename Tile, typename Out, typename In>
__device__ __forceinline__ void store_hwc(Tile *tile, const Hwc2HwcChwSampleDesc<Out, In> sample) {
  int64_t start_x = static_cast<int64_t>(blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);

  const auto *__restrict__ fill_values = static_cast<const Out *>(sample.fill_values);

  // Preload the norm values so they are accessed via registers and not from gmem via pointer.
  Compute norm_mul[kStaticChannels], norm_add[kStaticChannels];

#pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

  // Assuming all samples are padded
  if constexpr (enable_pad) {
    constexpr int kOutChannels = kStaticChannels + 1;
    int64_t block_4 = (kBlockSize / kStaticChannels) * kOutChannels;
    int64_t sample_size_4 = (sample.sample_size / kStaticChannels) * kOutChannels;
    int64_t start_x_padded = static_cast<int64_t>(blockIdx.x - sample.first_block) * block_4;
    int64_t end_x_padded = ::min(start_x_padded + block_4, sample_size_4);

    for (int64_t idx = threadIdx.x + start_x_padded, base_x = threadIdx.x; idx < end_x_padded;
         idx += blockDim.x, base_x += blockDim.x) {
      int base_offset = base_x >> 2;
      int c = idx & 3;

      int64_t out_offset = calculate_offset_hwc<enable_mirror, enable_pad>(idx, c, sample);

      if (c < kStaticChannels) {
        Compute fpin = tile[base_offset * sample.input_C + c];
        Compute fpout = fma(fpin, norm_mul[c], norm_add[c]);
        sample.out[out_offset] = ConvertSat<Out>(fpout);
      } else {
        sample.out[out_offset] = fill_values[c];
      }
    }
  } else {
    // No padding, we just with the same offset (or mirrored x offset)
    fast_div<uint32_t> channels(kStaticChannels);
    for (int64_t idx = threadIdx.x + start_x, base_x = threadIdx.x; idx < end_x;
         idx += blockDim.x, base_x += blockDim.x) {
      int c = idx % channels;

      int64_t out_offset = calculate_offset_hwc<enable_mirror, enable_pad>(idx, c, sample);

      Compute fpin = tile[base_x];
      Compute fpout = fma(fpin, norm_mul[c], norm_add[c]);
      sample.out[out_offset] = ConvertSat<Out>(fpout);
    }
  }
}

/**
 * @brief Store a tile of smem that is kept as planes in the HWC format.
 *
 * This version is specialized for uint8_t inputs and fp16 outputs + padding from 3 to 4 channels.
 * The output samples are expected to be aligned to at least 4-bytes allowing for vectorized
 * stores of __half2.
 * @tparam Compute Type to conduct computations in.
 * TODO(klecki): vectorized __half2 can be considered, float is ok.
 * @tparam Tile smem tile storage type
 */
template <int kBlockSize, int kStaticChannels, bool enable_mirror, typename Compute, typename Tile>
__device__ __forceinline__ void store_planar_hwc_pad(
    Tile tile[][kBlockSize / kStaticChannels],
    const Hwc2HwcChwSampleDesc<float16, uint8_t> sample) {
  constexpr int kOutChannels = kStaticChannels + 1;

  int64_t start_x = static_cast<int64_t>(blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);

  // Preload the norm values so they are accessed via registers and not from gmem via pointer.
  Compute norm_mul[kOutChannels], norm_add[kOutChannels];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

  // put the fill value so it will be produced as a result of FMA
  norm_mul[3] = 0;
  norm_add[3] = sample.fill_values[3];

  // Assuming all samples are padded
  int64_t block_4 = (kBlockSize / kStaticChannels) * kOutChannels;
  int64_t sample_size_4 = (sample.sample_size / kStaticChannels) * kOutChannels;
  int64_t start_x_padded = static_cast<int64_t>(blockIdx.x - sample.first_block) * block_4;
  int64_t end_x_padded = ::min(start_x_padded + block_4, sample_size_4);


  // TODO(klecki) in the version without mirror, we can keep one offset, as we can start the
  // output pointer at the output tile.
  auto *out_h2 = reinterpret_cast<__half2 *>(sample.out);
  uint32_t to_write = end_x_padded - start_x_padded;

  // loop is divided by two as we write two elements in each thread
  for (uint32_t base_x = threadIdx.x; base_x < to_write / 2; base_x += blockDim.x) {
    int base_offset = base_x / 2;
    int c = base_x & 1;

    int64_t out_offset;
    if constexpr (enable_mirror) {
      if (sample.flip_x) {
        int64_t idx = start_x_padded + base_x * 2;
        int y = idx / (sample.W * kOutChannels);
        int xc = idx - (int64_t)y * sample.W * kOutChannels;
        int x = xc / kOutChannels;
        int target_x = sample.W - 1 - x;
        // basically we divide the out_offset by two, The `c` is either 0 or 1.
        out_offset = (int64_t)y * sample.W * (kOutChannels / 2) + target_x * (kOutChannels / 2) + c;
      } else {
        out_offset = start_x_padded / 2 + base_x;
      }
    } else {
      out_offset = start_x_padded / 2 + base_x;
    }

    if (c == 0) {
      Compute fpin0 = tile[0][base_offset];
      Compute fpin1 = tile[1][base_offset];

      Compute fpout0 = fmaf(fpin0, norm_mul[0], norm_add[0]);
      Compute fpout1 = fmaf(fpin1, norm_mul[1], norm_add[1]);
      out_h2[out_offset] = make_half2(ConvertSat<float16>(fpout0), ConvertSat<float16>(fpout1));
    } else {
      Compute fpin0 = tile[2][base_offset];

      Compute fpout0 = fmaf(fpin0, norm_mul[2], norm_add[2]);
      // With more generic implementation, we could do the FMA for this value as well, but we
      // need to just pad it here.
      Compute fpout1 = norm_add[3];
      out_h2[out_offset] = make_half2(ConvertSat<float16>(fpout0), ConvertSat<float16>(fpout1));
    }
  }
}


/** @} */  // end of Hwc2HwcChwStore

/** @defgroup Hwc2HwcChw The Slice Hwc2{Hwc,Chw} Normalize Mirror-x Pad-channel kernel
 *
 * Kernel that reads a HWC u8 image and outputs a HWC or CHW normalized float image, that can be
 * cropped in Y, X coordinates, mirrored in X coordinate, and the channels can be padded.
 *
 * High level structure of the kernel:
 * 1. Load tile of linear data from the image into shared memory, doing a cast to floating type.
 *   a. Note, that the tile in shared memory can be represented either as an linear chunk with
 *      interleaved channels or as separate channel planes. See the loading functions for details.
 *   b. Each thread in loader loop maps to one value of the loaded image.
 *   c. Tile in shared memory doesn't take the padded channels into account, it stores only the
 *      input channels.
 * 2. Synchronize
 * 3. Output the data in correct layout, reading from the shared memory.
 *   a. For CHW output each thread corresponds to a (Y, X) sequential offset into a plane, computes
 *      the values for all the channels and writes them. Assuming 3-channel input, we can look
 *      at the input as a sequential stream of values, where we distribute them (sequentially)
 *      into 3 output planes.
 *   b. Padding the output channels for CHW is done by filling additional planes with fill values.
 *   c. For HWC output, in the simples case we can store the linear tile in the same order
 *      as it was read. In case of padding, fill values must be inserted.
 *   d. Mirroring is done by swapping the X-coordinate and recomputing the target offset for both
 *      layouts.
 *
 * The kernel use a thread block size, that is divisible both by channel number: 3 (for the
 * non-padded output loop), and 4 (alignment for input loop and padded output loop).
 *
 * For better throughput, the read and write accesses to global memory are sequential,
 * using aligned 4-byte-wide access when possible.
 * @{
 */

// TODO(klecki): generalize for wider input types

/**
 * @brief Hwc2HwcChw Normalize Mirror-x Pad-channel kernel
 * This kernel does not support cropping the x coordinate, so the reads are fully linear.
 */
template <typename Out, typename In, bool enable_mirror, bool enable_pad, int kBlockSize,
          int kStaticChannels>
__global__ void Hwc2HwcChwNormalize(const Hwc2HwcChwSampleDesc<Out, In> *samples,
                                    uint32_t *first_blocks, uint32_t num_samples) {
  static_assert(std::is_same<In, uint8_t>::value, "Only uint8_t supported as input");

  int sample_idx = FindSampleIdx(first_blocks, num_samples);
  const auto sample = samples[sample_idx];

  __shared__ float tile[kBlockSize + 32 * 4];

  float *loaded_tile = load_linear_tile<kBlockSize, kStaticChannels>(tile, sample);

  __syncthreads();

  store_chw<kBlockSize, kStaticChannels, enable_mirror, enable_pad>(loaded_tile, sample);
}

/**
 * @brief Slice Hwc2HwcChw Normalize [Mirror-x] [Pad-channel] kernel
 * This kernel supports cropping in x-coordinate.
 */
template <typename Out, typename In, bool enable_mirror, bool enable_pad, int kBlockSize,
          int kStaticChannels>
__global__ void SliceHwc2HwcChwNormalize(const Hwc2HwcChwSampleDesc<Out, In> *samples,
                                         uint32_t *first_blocks, uint32_t num_samples) {
  static_assert(std::is_same<In, uint8_t>::value, "Only uint8_t supported as input");

  int sample_idx = FindSampleIdx(first_blocks, num_samples);
  const auto sample = samples[sample_idx];

  __shared__ float tile[kBlockSize + 32 * 4];
  float *loaded_tile = slice_load_linear_tile<kBlockSize, kStaticChannels>(tile, sample);

  __syncthreads();

  store_chw<kBlockSize, kStaticChannels, enable_mirror, enable_pad>(loaded_tile, sample);
}

/**
 * @brief Hwc2Hwc Normalize [Mirror-x] [Pad-channel] kernel
 * This kernel does not support cropping the x coordinate, so the reads are fully linear.
 */
template <typename Out, typename In, bool enable_mirror, bool enable_pad, int kBlockSize,
          int kStaticChannels>
__global__ void Hwc2HwcNormalize(const Hwc2HwcChwSampleDesc<Out, In> *samples,
                                 uint32_t *first_blocks, uint32_t num_samples) {
  static_assert(std::is_same<In, uint8_t>::value, "Only uint8_t supported as input");

  int sample_idx = FindSampleIdx(first_blocks, num_samples);
  const auto sample = samples[sample_idx];

  __shared__ float tile[kBlockSize + 32 * 4];
  float *loaded_tile = load_linear_tile<kBlockSize, kStaticChannels>(tile, sample);

  __syncthreads();

  store_hwc<kBlockSize, kStaticChannels, enable_mirror, enable_pad, Out>(loaded_tile, sample);
}

/**
 * @brief Slice Hwc2Hwc Normalize [Mirror-x] [Pad-channel] kernel
 * This kernel supports cropping in x-coordinate.
 */
template <typename Out, typename In, bool enable_mirror, bool enable_pad, int kBlockSize,
          int kStaticChannels>
__global__ void SliceHwc2HwcNormalize(const Hwc2HwcChwSampleDesc<Out, In> *samples,
                                      uint32_t *first_blocks, uint32_t num_samples) {
  static_assert(std::is_same<In, uint8_t>::value, "Only uint8_t supported as input");

  int sample_idx = FindSampleIdx(first_blocks, num_samples);
  const auto sample = samples[sample_idx];

  __shared__ float tile[kBlockSize + 32 * 4];
  float *loaded_tile = slice_load_linear_tile<kBlockSize, kStaticChannels>(tile, sample);

  __syncthreads();

  store_hwc<kBlockSize, kStaticChannels, enable_mirror, enable_pad, Out>(loaded_tile, sample);
}

/**
 * @brief Hwc2Hwc Normalize [Mirror-x] Pad-channel-always kernel for FP16.
 *
 * This kernel utilizes 4-byte reads and writes. The smem intermediate tile uses planar layout,
 * for better access to the image values during writing of the output.
 * The output samples are assumed to be aligned to the address that is multiple of 4,
 * thanks to the padding performed to 4 channels, it holds for every batch that is laid out
 * contiguously in memory with aligned start. This holds for forseeable future in DALI.
 */
template <typename Out, typename In, bool enable_mirror, int kBlockSize, int kStaticChannels>
__global__ void Hwc2HwcNormalizePadFp16(const Hwc2HwcChwSampleDesc<Out, In> *samples,
                                        uint32_t *first_blocks, uint32_t num_samples) {
  static_assert(std::is_same<In, uint8_t>::value, "Only uint8_t supported as input");

  int sample_idx = FindSampleIdx(first_blocks, num_samples);
  const auto sample = samples[sample_idx];

  __shared__ float tile[kStaticChannels][kBlockSize / kStaticChannels];
  load_planar_tile<kBlockSize, kStaticChannels>(tile, sample);

  __syncthreads();

  store_planar_hwc_pad<kBlockSize, kStaticChannels, enable_mirror, float>(tile, sample);
}


/** @} */  // end of Hwc2HwcChw

template <typename Out>
KernelRequirements SliceHwc2HwcChwNormalizeGPU<Out>::Setup(KernelContext &ctx,
                                                           const TensorListShape<ndim> &input_shape,
                                                           span<const SampleArgs> args,
                                                           TensorLayout output_layout) {
  (void)ctx;
  int num_samples = input_shape.num_samples();
  DALI_ENFORCE(num_samples == static_cast<int>(args.size()),
               "Invalid number of samples in kernel args");
  out_shape_ = TensorListShape<ndim>(num_samples, ndim);
  collapsed_tiling_shape_ = TensorListShape<1>(num_samples, 1);

  perm_ = output_layout == "HWC" ? std::array<int, 3>{0, 1, 2} : std::array<int, 3>{2, 0, 1};
  output_layout_ = output_layout;

  SetupNumChannels(input_shape, args);
  DALI_ENFORCE(output_layout == "HWC" || output_layout == "CHW",
               "Only CHW and HWC output layouts allowed");

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
std::tuple<float *, float *, Out *> SliceHwc2HwcChwNormalizeGPU<Out>::SetupParams(
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
auto SliceHwc2HwcChwNormalizeGPU<Out>::RealignSample(
    TensorView<StorageGPU, const In, ndim> in_sample, Roi<spatial_dim> roi)
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
void SliceHwc2HwcChwNormalizeGPU<Out>::SetupNumChannels(const TensorListShape<ndim> &input_shape,
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
      "Number of samples in the arguments should match the number of samples in the shape.");

  out_nchannels_ = std::max(nchannels_, static_cast<int>(args[0].fill_values.size()));
  for (int i = 1; i < input_shape.num_samples(); i++) {
    DALI_ENFORCE(args[i].fill_values.size() == args[0].fill_values.size(),
                 "All sample arguments should have the same number of fill values.");
  }
  DALI_ENFORCE(nchannels_ == kStaticChannels, "Only 3 input channels are supported.");
  if (output_layout_ == "HWC") {
    // Padding in the operator cannot go higher than the closest power of 2,
    // but better have the check in place.
    DALI_ENFORCE(out_nchannels_ == kStaticChannels || out_nchannels_ == kStaticChannels + 1,
                 "Only 3 or 4 output channels are supported for HWC output layout.");
  }
}


template <typename Out>
void SliceHwc2HwcChwNormalizeGPU<Out>::Run(KernelContext &ctx,
                                           const TensorListView<StorageGPU, Out, ndim> &out,
                                           const TensorListView<StorageGPU, const In, ndim> &in,
                                           span<const SampleArgs> args) {
  using SampleDesc = Hwc2HwcChwSampleDesc<Out, In>;
  int num_samples = in.num_samples();

  SampleDesc *sample_descs_cpu = ctx.scratchpad->AllocatePinned<SampleDesc>(num_samples);
  uint32_t *first_blocks_cpu = ctx.scratchpad->AllocatePinned<uint32_t>(num_samples);
  auto [norm_add_gpu, norm_mul_gpu, fill_values_gpu] = SetupParams(ctx, args);
  bool need_pad = out_nchannels_ != nchannels_;
  bool need_crop_x = false;
  bool need_flip_x = false;
  // Check if all the outputs are aligned to 4 bytes, used by the specialized FP16 PAD HWC -> HWC
  // implementation. With the current state of DALI, the start of output allocation is aligned
  // (to even higher power of two), and all the samples have length that is multiple of 4 (padded to
  // 4 channels), that is if they are in contiguous allocation, all output samples are still aligned
  // to a multiple of 4.
  bool outputs_aligned_4 = true;

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
    if (reinterpret_cast<std::uintptr_t>(sample_desc.out) % 4) {
      outputs_aligned_4 = false;
    }

    first_block = offset_blk;
    sample_desc.first_block = offset_blk;
    sample_desc.sample_size = sample_size;
    offset_blk += div_ceil(sample_size, kBlockSizeMul * kBlockWidth);

    // The output shape here is after the permutation
    if (output_layout_ == "CHW") {
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

  auto [sample_descs_gpu, first_blocks_gpu] =
      ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, make_span(sample_descs_cpu, nonempty_samples),
                                      make_span(first_blocks_cpu, nonempty_samples));

  // TODO(klecki): Maybe this selection can be simplified, but making the output layout
  // a parameter would probably make it even less readable.
  // This version allows utilizing specialized implementations for every layout more easily.
  if (output_layout_ == "CHW") {
    auto dispatch = [samples = sample_descs_gpu, blocks = first_blocks_gpu, &ctx, need_crop_x,
                     offset_blk, nonempty_samples](auto pad_v, auto flip_x_v) {
      if (need_crop_x) {
        SliceHwc2HwcChwNormalize<Out, In, flip_x_v.value, pad_v.value, kBlockSizeMul * kBlockWidth,
                                 kStaticChannels>
            <<<offset_blk, kThreadBlockSize, 0, ctx.gpu.stream>>>(samples, blocks,
                                                                  nonempty_samples);
      } else {
        Hwc2HwcChwNormalize<Out, In, flip_x_v.value, pad_v.value, kBlockSizeMul * kBlockWidth,
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
    auto dispatch = [samples = sample_descs_gpu, blocks = first_blocks_gpu, &ctx, need_crop_x,
                     offset_blk, nonempty_samples](auto pad_v, auto flip_x_v, auto out_aligned_v) {
      if (need_crop_x) {
        SliceHwc2HwcNormalize<Out, In, flip_x_v.value, pad_v.value, kBlockSizeMul * kBlockWidth,
                              kStaticChannels><<<offset_blk, kThreadBlockSize, 0, ctx.gpu.stream>>>(
            samples, blocks, nonempty_samples);
      } else {
        if constexpr (std::is_same_v<Out, float16> && pad_v.value && out_aligned_v.value) {
          Hwc2HwcNormalizePadFp16<Out, In, flip_x_v.value, kBlockSizeMul * kBlockWidth,
                                  kStaticChannels>
              <<<offset_blk, kThreadBlockSize, 0, ctx.gpu.stream>>>(samples, blocks,
                                                                    nonempty_samples);
        } else {
          Hwc2HwcNormalize<Out, In, flip_x_v.value, pad_v.value, kBlockSizeMul * kBlockWidth,
                           kStaticChannels><<<offset_blk, kThreadBlockSize, 0, ctx.gpu.stream>>>(
              samples, blocks, nonempty_samples);
        }
      }
    };

     auto dispatch_aligned = [&](auto pad_v, auto flip_x_v, bool out_aligned) {
      if (out_aligned) {
        dispatch(pad_v, flip_x_v, std::true_type{});
      } else {
        dispatch(pad_v, flip_x_v, std::false_type{});
      }
    };

    auto dispatch_flip = [&](auto pad_v, bool flip_x, bool out_aligned) {
      if (flip_x) {
        dispatch_aligned(pad_v, std::true_type{}, out_aligned);
      } else {
        dispatch_aligned(pad_v, std::false_type{}, out_aligned);
      }
    };

    if (need_pad) {
      dispatch_flip(std::true_type{}, need_flip_x, outputs_aligned_4);
    } else {
      dispatch_flip(std::false_type{}, need_flip_x, outputs_aligned_4);
    }
  }

  CUDA_CALL(cudaGetLastError());
}


template class DLL_PUBLIC SliceHwc2HwcChwNormalizeGPU<float>;
template class DLL_PUBLIC SliceHwc2HwcChwNormalizeGPU<float16>;

}  // namespace slice_flip_normalize

}  // namespace kernels
}  // namespace dali
