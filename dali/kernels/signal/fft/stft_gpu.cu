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

#include "dali/kernels/signal/fft/stft_gpu.h"
#include "dali/kernels/signal/fft/stft_gpu_impl.cuh"
#include <memory>

namespace dali {
namespace kernels {
namespace signal {
namespace fft {

StftGPU::StftGPU() = default;
StftGPU::StftGPU(StftGPU &&) = default;
StftGPU::~StftGPU() = default;

kernels::KernelRequirements StftGPU::Setup(
    KernelContext &ctx,
    const TensorListShape<1> &in_shape,
    const StftArgs &args) {
  if (!impl_)
    impl_ = std::make_unique<StftImplGPU>();
  DALI_ENFORCE(args.spectrum_type == FFT_SPECTRUM_COMPLEX);
  return impl_->Setup(ctx, in_shape, args);
}

void StftGPU::Run(
    KernelContext &ctx,
    const OutListGPU<complexf, 2> &out,
    const InListGPU<float, 1> &in,
    const InTensorGPU<float, 1> &window) {
  assert(impl_ != nullptr && "No instance present - missing call to Setup?");
  impl_->Run(ctx, out, in, window);
}

SpectrogramGPU::SpectrogramGPU() = default;
SpectrogramGPU::SpectrogramGPU(SpectrogramGPU &&) = default;
SpectrogramGPU::~SpectrogramGPU() = default;

kernels::KernelRequirements SpectrogramGPU::Setup(
    KernelContext &ctx,
    const TensorListShape<1> &in_shape,
    const StftArgs &args) {
  if (!impl_)
    impl_ = std::make_unique<StftImplGPU>();
  DALI_ENFORCE(args.spectrum_type != FFT_SPECTRUM_COMPLEX);
  return impl_->Setup(ctx, in_shape, args);
}

void SpectrogramGPU::Run(
    KernelContext &ctx,
    const OutListGPU<float, 2> &out,
    const InListGPU<float, 1> &in,
    const InTensorGPU<float, 1> &window) {
  assert(impl_ != nullptr && "No instance present - missing call to Setup?");
  impl_->Run(ctx, out, in, window);
}

}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali
