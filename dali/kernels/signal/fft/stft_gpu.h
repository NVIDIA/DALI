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

#ifndef DALI_KERNELS_SIGNAL_FFT_STFT_GPU_H_
#define DALI_KERNELS_SIGNAL_FFT_STFT_GPU_H_

#include <memory>
#include "dali/core/tensor_view.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/signal/fft/fft_common.h"
#include "dali/kernels/signal/window/extract_windows_args.h"

namespace dali {
namespace kernels {
namespace signal {
namespace fft {

struct StftArgs : ExtractWindowsArgs {
  FftSpectrumType spectrum_type = FFT_SPECTRUM_POWER;

  int nfft = -1;

  bool time_major_layout = false;

  DALI_HOST_DEV
  constexpr inline bool operator==(const StftArgs &other) const {
    return ExtractWindowsArgs::operator==(other) &&
           nfft == other.nfft &&
           time_major_layout == other.time_major_layout &&
           spectrum_type == other.spectrum_type;
  }

  DALI_HOST_DEV
  constexpr inline bool operator!=(const StftArgs &other) const {
    return !(*this == other);
  }
};

class StftImplGPU;

class DLL_PUBLIC StftGPU {
 public:
  StftGPU();
  StftGPU(StftGPU &&);
  ~StftGPU();
  kernels::KernelRequirements Setup(
    KernelContext &ctx,
    const TensorListShape<1> &in,
    const StftArgs &args);

  void Run(
    KernelContext &ctx,
    const OutListGPU<complexf, 2> &out,
    const InListGPU<float, 1> &in,
    const InTensorGPU<float, 1> &window);

 private:
  std::unique_ptr<StftImplGPU> impl_;
};

class DLL_PUBLIC SpectrogramGPU {
 public:
  SpectrogramGPU();
  SpectrogramGPU(SpectrogramGPU &&);
  ~SpectrogramGPU();
  kernels::KernelRequirements Setup(
    KernelContext &ctx,
    const TensorListShape<1> &in,
    const StftArgs &args);

  void Run(
    KernelContext &ctx,
    const OutListGPU<float, 2> &out,
    const InListGPU<float, 1> &in,
    const InTensorGPU<float, 1> &window);

 private:
  std::unique_ptr<StftImplGPU> impl_;
};

}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali


#endif  // DALI_KERNELS_SIGNAL_FFT_STFT_GPU_H_
