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

#ifndef DALI_KERNELS_AUDIO_MEL_SCALE_MEL_FILTER_BANK_CPU_H_
#define DALI_KERNELS_AUDIO_MEL_SCALE_MEL_FILTER_BANK_CPU_H_

#include <memory>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/audio/mel_scale/mel_filter_bank_args.h"

namespace dali {
namespace kernels {
namespace audio {

template <typename T = float>
class DLL_PUBLIC MelFilterBankCpu {
 public:
  static_assert(std::is_floating_point<T>::value, "Only floating point types are supported");

  DLL_PUBLIC MelFilterBankCpu();
  DLL_PUBLIC ~MelFilterBankCpu();

  DLL_PUBLIC KernelRequirements Setup(KernelContext &context,
                                      const InTensorCPU<T> &in,
                                      const MelFilterBankArgs &args);

  DLL_PUBLIC void Run(KernelContext &context,
                      const OutTensorCPU<T> &out,
                      const InTensorCPU<T> &in);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace audio
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_AUDIO_MEL_SCALE_MEL_FILTER_BANK_CPU_H_
