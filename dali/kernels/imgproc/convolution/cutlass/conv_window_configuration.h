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

template <int TotalAlignedSize, bool IsInnerConv, bool UseSharedMem = true>
struct ConvWindowConfiguration {
  static int const kTotalAlignedSize = TotalAlignedSize;
  static bool const kIsInnerConv = IsInnerConv;
  static bool const kUseSharedMem = UseSharedMem;
  static_assert(kUseSharedMem, "Reading window directly from GMEM is not yet implemented");
  static_assert((!kIsInnerConv && kTotalAlignedSize % 4 == 0) ||
                    (kIsInnerConv && kTotalAlignedSize % 2 == 0),
                "The total window size needs to be divisible for alignment purposes");
  static int const kWindowDecreasingCenter =
      kIsInnerConv ? kTotalAlignedSize / 2 : kTotalAlignedSize / 4;
  static int const kWindowIncreasingCenter = kIsInnerConv ? -1 : (kTotalAlignedSize / 4) * 3;

  template <typename T>
  using PaddedWindowBuffer = dali::span<T, ConvWindowConfiguration::kTotalAlignedSize>;

  template <bool mirrored>
  CUTLASS_HOST_DEVICE constexpr static int getWindowCenter() {
    return mirrored ? kWindowDecreasingCenter : kWindowIncreasingCenter;
  }

  static int const kMaxWindowRadiusSpan =
      kIsInnerConv ? kTotalAlignedSize / 2 : kTotalAlignedSize / 4;

  /**
   * @brief Layouts the window with the padding required by the PositionPredicatedTileIterator
   */
  template <typename T>
  static void prepare_window(PaddedWindowBuffer<T> dst, dali::span<const T> src,
                             int num_channels = 1) {
    int radius = src.size() / 2;
    memset(dst.data(), 0, sizeof(T) * kTotalAlignedSize);
    for (int i = 0; i <= radius; i++) {
      if (kIsInnerConv) {
        dst[kWindowDecreasingCenter + num_channels * i] = src[radius - i];
        dst[kWindowDecreasingCenter - num_channels * i] = src[radius + i];
      } else {
        dst[kWindowIncreasingCenter - i] = src[radius - i];
        dst[kWindowIncreasingCenter + i] = src[radius + i];

        dst[kWindowDecreasingCenter + i] = src[radius - i];
        dst[kWindowDecreasingCenter - i] = src[radius + i];
      }
    }
  }
};

}  // namespace gemm
}  // namespace cutlass

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_CONV_WINDOW_CONFIGURATION_H_
