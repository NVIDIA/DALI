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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_CONV_WINDOW_CONFIGURATION_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_CONV_WINDOW_CONFIGURATION_H_

#include "cutlass/cutlass.h"

#include "dali/core/span.h"

namespace cutlass {
namespace gemm {

template <int TotalAlignedSize, bool IsInnerConv, int AccessSize = 1, bool UseSharedMem = true>
struct ConvWindowConfiguration {
  static int const kTotalAlignedSize = TotalAlignedSize;
  static bool const kIsInnerConv = IsInnerConv;
  static bool const kUseSharedMem = UseSharedMem;
  static int const kAccessSize = AccessSize;
  static_assert(kUseSharedMem, "Reading window directly from global memory is not yet implemented");

  // For inner convolution we only need a reversed window (decreasing order)
  // For outer convolution we need both the regular window (increasing order) and the reversed
  // window (decreasing order). For details, see the PositionPredicatedTileIterator.
  static_assert((!kIsInnerConv && kTotalAlignedSize % 4 == 0) ||
                    (kIsInnerConv && kTotalAlignedSize % 2 == 0),
                "The total window size needs to be divisible for alignment purposes");

  static int const kWindowDecreasingStart =
      kIsInnerConv ? kAccessSize : kTotalAlignedSize / 2 + kAccessSize;
  static int const kWindowIncreasingStart = kIsInnerConv ? -1 : kAccessSize;

  template <typename T>
  using PaddedWindowBuffer = dali::span<T, ConvWindowConfiguration::kTotalAlignedSize>;

  template <bool mirrored>
  CUTLASS_HOST_DEVICE constexpr static int getWindowStart() {
    return mirrored ? kWindowDecreasingStart : kWindowIncreasingStart;
  }

  static int const kMaxWindowSize =
      kIsInnerConv ? kTotalAlignedSize - 2 * AccessSize : kTotalAlignedSize / 2 - 2 * kAccessSize;

  /**
   * @brief Layouts the window with the padding required by the PositionPredicatedTileIterator
   */
  template <typename T>
  static void prepare_window(PaddedWindowBuffer<T> dst, dali::span<const T> src,
                             int num_channels = 1) {
    int window_size = src.size();
    memset(dst.data(), 0, sizeof(T) * kTotalAlignedSize);
    // As the PositionPredicatedTileIterator handles accesses like in case of correllation
    // and not convolution, we have additional window-flip here
    if (kIsInnerConv) {
      for (int i = 0; i < window_size; i++) {
        dst[kWindowDecreasingStart + num_channels * i] = src[i];
      }
    } else {
      for (int i = 0; i < window_size; i++) {
        dst[kWindowIncreasingStart + i] = src[window_size - i - 1];
      }
      for (int i = 0; i < window_size; i++) {
        dst[kWindowDecreasingStart + i] = src[i];
      }
    }
  }
};

}  // namespace gemm
}  // namespace cutlass

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_CONV_WINDOW_CONFIGURATION_H_
