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

#ifndef DALI_KERNELS_SIGNAL_FFT_STFT_GPU_H_
#define DALI_KERNELS_SIGNAL_FFT_STFT_GPU_H_

#include <memory>
#include "dali/core/tensor_view.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/signal/window/extract_windows_args.h"

namespace dali {
namespace kernels {
namespace signal {
namespace fft {

struct StftArgs : ExtractWindowsArgs {
  int power = 2;  // 1 = magnitude, 2 = squared magnitude

  /// @brief If set to > 0, the implementation setup will try to provide optimum
  ///        environment for processing batches of this combined length.
  int64_t estimated_max_total_length = 0;
};

class StftGpuImpl;

class DLL_PUBLIC StftGpu {
 public:
  ~StftGpu();
  kernels::KernelRequirements Setup(
    KernelContext &ctx,
    const InListGPU<float, 1> &in,
    const InTensorGPU<float, 1> &window,
    const StftArgs &args);

  void Run(
    KernelContext &ctx,
    const OutListGPU<float, 2> &out,
    const InListGPU<float, 1> &in,
    const InTensorGPU<float, 1> &window,
    const StftArgs &args);

 private:
  std::unique_ptr<StftGpuImpl> impl;
};

}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali


#endif  // DALI_KERNELS_SIGNAL_FFT_STFT_GPU_H_
