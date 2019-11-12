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

#include "dali/kernels/signal/window/extract_frames_cpu.h"
#include <algorithm>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {
namespace signal {
namespace window {

template <typename OutputType, typename InputType, int Dims>
ExtractFramesCpu<OutputType, InputType, Dims>::~ExtractFramesCpu() = default;

template <typename OutputType, typename InputType, int Dims>
KernelRequirements ExtractFramesCpu<OutputType, InputType, Dims>::Setup(
    KernelContext &context,
    const InTensorCPU<InputType, InputDims> &in,
    const ExtractFramesArgs &args) {
  KernelRequirements req;
  DALI_FAIL("not implemented");

  constexpr int64_t kMinWindowLength = 1;
  window_length_ = std::max(args.window_length, kMinWindowLength);
  constexpr int64_t kMinWindowStep = 1;
  window_step_ = std::max(args.window_step, kMinWindowStep);

  if (args.axis < 0) {
    axis_ = Dims - 1;
  }

  const auto n = in.shape[axis_];

  nwindows_ = (n + window_step_ - 1) / window_step_;
  assert(nwindows_ > 0);

  return req;
}

template <typename OutputType, typename InputType, int Dims>
void ExtractFramesCpu<OutputType, InputType, Dims>::Run(
    KernelContext &context,
    const OutTensorCPU<OutputType, OutputDims> &out,
    const InTensorCPU<InputType, InputDims> &in,
    const ExtractFramesArgs &args) {
  DALI_FAIL("not implemented");
}

template class ExtractFramesCpu<float, float, 1>;  // 1-channel
template class ExtractFramesCpu<uint8_t, uint8_t, 1>;  // 1-channel

template class ExtractFramesCpu<float, float, 2>;  // n-channel
template class ExtractFramesCpu<uint8_t, uint8_t, 2>;  // n-channel


}  // namespace window
}  // namespace signal
}  // namespace kernels
}  // namespace dali
