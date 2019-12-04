// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_SIGNAL_WINDOW_EXTRACT_WINDOWS_GPU_H_
#define DALI_KERNELS_SIGNAL_WINDOW_EXTRACT_WINDOWS_GPU_H_

#include <memory>
#include "dali/core/convert.h"
#include "dali/kernels/signal/window/extract_windows_args.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {
namespace signal {

struct ExtractWindowsBatchedArgs : ExtractWindowsArgs {
  /**
   * @brief If true, all outputs are concatenated.
   *
   * The concatenated output will contain all first samples from windows from 1st recording,
   * then 2nd, etc, and then all seconds samples from all recordings and so forth.
   */
  bool concatenate = true;
  /**
   * @brief Indicates that the output should be overallocated (or windows truncated) to this size.
   */
  int padded_output_window = -1;
};


/**
 * @brief Extracts windows from 1D signal, optionally applying a custom window function
 *
 * This kernel copies evenly spaced 1D windows from the input signal(s) to the output.
 * The windows can (and usually do) overlap. If the `window` argument is non-empty, it's used
 * as a window function - i.e. its pointwise multiplied with each extracted window element
 * before storing it to the output.
 *
 * The output windows are stored as columns - there's a row of first samples from all windows,
 * then second samples, etc - a layout typically used in spectrograms.
 * If the `ExtractWindowsBatchedArgs::concatenate` is true, then the output rows are concatenated and
 * the output TensorList contains just one tensor.
 *
 * @see ExtractWindowsBatchedArgs
 * @see ExtractWindowsArgs
 */
template <typename Dst, typename Src>
class DLL_PUBLIC ExtractWindowsGpu {
 public:
  static_assert(std::is_same<Dst, float>::value, "Output type must be float");
  static_assert(
    std::is_same<Src, float>::value ||
    std::is_same<Src, int8_t>::value ||
    std::is_same<Src, int16_t>::value, "Input type must be float, int8_t or int16_t");

  DLL_PUBLIC KernelRequirements Setup(
      KernelContext &context,
      const InListGPU<Src, 1> &input,
      const InTensorGPU<float, 1> &window,
      const ExtractWindowsBatchedArgs &args);

  DLL_PUBLIC void Run(KernelContext &ctx,
      const OutListGPU<Dst, 2> &out,
      const InListGPU<Src, 1> &in,
      const InTensorGPU<float, 1> &window,
      const ExtractWindowsBatchedArgs &args);

  DLL_PUBLIC virtual ~ExtractWindowsGpu();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_WINDOW_EXTRACT_WINDOWS_GPU_H_
