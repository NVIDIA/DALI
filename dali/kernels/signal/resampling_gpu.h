// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_SIGNAL_RESAMPLING_GPU_H_
#define DALI_KERNELS_SIGNAL_RESAMPLING_GPU_H_

#include <cuda_runtime.h>
#include "dali/kernels/signal/resampling.h"
#include "dali/kernels/kernel.h"
#include "dali/core/dev_buffer.h"

namespace dali {
namespace kernels {
namespace signal {

namespace resampling {

template <typename Out = float, typename In = Out>
class DLL_PUBLIC ResamplerGPU {
 public:
  void Initialize(int lobes = 16, int lookup_size = 2048);

  KernelRequirements Setup(KernelContext &context, const InListGPU<In> &in, span<const Args> args);

  void Run(KernelContext &context, const OutListGPU<Out> &out,
           const InListGPU<In> &in, span<const Args> args);

 private:
  ResamplingWindowCPU window_cpu_;
  ResamplingWindow window_gpu_;
  DeviceBuffer<float> window_gpu_storage_;
};

#define DALI_INSTANTIATE_RESAMPLER_GPU_OUT(linkage, Out)\
  linkage template class ResamplerGPU<Out, float>; \
  linkage template class ResamplerGPU<Out, int8_t>; \
  linkage template class ResamplerGPU<Out, uint8_t>; \
  linkage template class ResamplerGPU<Out, int16_t>; \
  linkage template class ResamplerGPU<Out, uint16_t>; \
  linkage template class ResamplerGPU<Out, int32_t>; \
  linkage template class ResamplerGPU<Out, uint32_t>; \

#define DALI_INSTANTIATE_RESAMPLER_GPU(linkage) \
  DALI_INSTANTIATE_RESAMPLER_GPU_OUT(linkage, float) \
  DALI_INSTANTIATE_RESAMPLER_GPU_OUT(linkage, int8_t) \
  DALI_INSTANTIATE_RESAMPLER_GPU_OUT(linkage, uint8_t) \
  DALI_INSTANTIATE_RESAMPLER_GPU_OUT(linkage, int16_t) \
  DALI_INSTANTIATE_RESAMPLER_GPU_OUT(linkage, uint16_t) \
  DALI_INSTANTIATE_RESAMPLER_GPU_OUT(linkage, int32_t) \
  DALI_INSTANTIATE_RESAMPLER_GPU_OUT(linkage, uint32_t)

DALI_INSTANTIATE_RESAMPLER_GPU(extern)

}  // namespace resampling
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_RESAMPLING_GPU_H_
