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

#ifndef DALI_KERNELS_AUDIO_FFT_FFT_CPU_H_
#define DALI_KERNELS_AUDIO_FFT_FFT_CPU_H_

#include "dali/core/format.h"
#include "dali/core/common.h"
#include "dali/core/util.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include <ffts/ffts.h>

namespace dali {
namespace kernels {
namespace audio {
namespace fft {

enum FftOutputType {
  FFT_OUTPUT_TYPE_COMPLEX = 0,    // separate interleaved real and img parts: (r0, i0, r1, i2, ...)
  FFT_OUTPUT_TYPE_MAGNITUDE = 1,  // (only forward) sqrt( real^2 + img^2 )
  FFT_OUTPUT_TYPE_POWER = 2,      // (only forward) real^2 + img^2
};

struct FftArgs {
  FftOutputType output_type = FFT_OUTPUT_TYPE_COMPLEX;
  int transform_axis = 0;
  int channel_axis = 1;
};

template <typename OutputType, typename InputType = OutputType, int Dims = 2>
class DLL_PUBLIC Fft1DCpu {
 public:
  static_assert(std::is_same<InputType, OutputType>::value
             && std::is_same<InputType, float>::value,
    "Data types other than float are not yet supported");

  static_assert(Dims == 2,
    "Expected 2D data where dim 0 represents the sample space and dim 1 the different channels");

  DLL_PUBLIC KernelRequirements Setup(KernelContext &context,
                                      const InTensorCPU<InputType, Dims> &in,
                                      const FftArgs &args);

  DLL_PUBLIC void Run(KernelContext &context,
                      OutTensorCPU<OutputType, Dims> &out,
                      const InTensorCPU<InputType, Dims> &in,
                      const FftArgs &args);
 private:
  void ValidateArgs(const FftArgs& args);

  using FftsPlanPtr = std::unique_ptr<ffts_plan_t, decltype(&ffts_free)>;
  FftsPlanPtr plan_{nullptr, ffts_free};
  int64_t plan_n_ = -1;
};

}  // namespace fft
}  // namespace audio
}  // namespace kernels
}  // namespace dali

  #endif  // DALI_KERNELS_AUDIO_FFT_FFT_CPU_H_