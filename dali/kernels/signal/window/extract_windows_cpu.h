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

#ifndef DALI_KERNELS_SIGNAL_WINDOW_EXTRACT_WINDOWS_CPU_H_
#define DALI_KERNELS_SIGNAL_WINDOW_EXTRACT_WINDOWS_CPU_H_

#include <memory>
#include <complex>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/signal/window/extract_windows_args.h"

namespace dali {
namespace kernels {
namespace signal {

/**
 * @brief Extracts windows from 1D signal applying a custom window function
 *
 * @tparam OutputType Output data type
 * @tparam InputType Input data type
 * @tparam Dims Number of dimensions
 * @tparam vertical If true, the window index dimension comes right after the temporal dimension,
 *         resulting in "vertical" windows in the output layout.
 *         Otherwise, the new window index dimension is placed just before the time dimension,
 *         producing "horizontal windows" in the output layout.
 * @param args.window_length Window size in number of samples
 *
 * @param args.window_step Length of the step between windows. If not provided, win_length
 *        will be choosen (no overlap)
 *
 * @param args.axis Index of the axis representing the temporal dimension. If not provided,
 *        the last dimension will be used
 *
 * @param args.window_center If set, window centers will be adjusted with this offset.
 *        Note: window_center must be a value within 0 and window_length
 *        If not specified, window_center = window_length / 2 will be assumed
 *        With window_center = window_length / 2, window centers are placed at multiples of
 *          `window_step`.
 *        With window_center = 0, windows are aligned to the left (window start at multiples of
 *          `window_step`).
 *        With window_center = window_length, windows are aligned to the right (window start at
 *          multiples of `window_step`).
 *
 * @param args.reflect_pad Determines the padding policy when sampling out of bounds.
 *        If true, the signal will be padded with its own reflection.
 *        If false, the signal will be padded with zeros.
 *        This option is only relevant when `window_center` is greater than 0
 *
 */
template <typename OutputType = float, typename InputType = float, int Dims = 1,
          bool vertical = true>
class DLL_PUBLIC ExtractWindowsCpu {
 public:
  static constexpr int InputDims = Dims;
  static constexpr int OutputDims = Dims + 1;

  static_assert(std::is_same<OutputType, InputType>::value,
    "Type conversion is not allowed in this kernel");

  DLL_PUBLIC ~ExtractWindowsCpu();

  DLL_PUBLIC KernelRequirements Setup(KernelContext &context,
                                      const InTensorCPU<InputType, InputDims> &in,
                                      const InTensorCPU<float, 1> &window_fn,
                                      const ExtractWindowsArgs &args);

  DLL_PUBLIC void Run(KernelContext &context,
                      const OutTensorCPU<OutputType, OutputDims> &out,
                      const InTensorCPU<InputType, InputDims> &in,
                      const InTensorCPU<float, 1> &window_fn,
                      const ExtractWindowsArgs &args);

 private:
  int window_length_ = -1;
  int window_step_ = -1;
  int window_fn_length_ = -1;
  int axis_ = -1;
  int window_center_offset_ = 0;
  int64_t nwindows_ = -1;
  Padding padding_ = Padding::Zero;
};

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_WINDOW_EXTRACT_WINDOWS_CPU_H_
