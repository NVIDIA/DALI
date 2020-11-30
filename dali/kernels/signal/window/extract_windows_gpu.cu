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

#include <memory>
#include "dali/kernels/signal/window/extract_windows_gpu.h"
#include "dali/kernels/signal/window/extract_windows_gpu.cuh"

namespace dali {
namespace kernels {
namespace signal {

template <typename Dst, typename Src>
KernelRequirements ExtractWindowsGPU<Dst, Src>::Setup(
    KernelContext &context,
    const InListGPU<Src, 1> &input,
    const InTensorGPU<float, 1> &window,
    const ExtractWindowsBatchedArgs &args) {
  return Setup(context, make_span(input.shape.shapes), args);
}

template <typename Dst, typename Src>
KernelRequirements ExtractWindowsGPU<Dst, Src>::Setup(
    KernelContext &context,
    const TensorListShape<1> &input_shape,
    const ExtractWindowsBatchedArgs &args) {
  return Setup(context, make_span(input_shape.shapes), args);
}

template <typename Dst, typename Src>
KernelRequirements ExtractWindowsGPU<Dst, Src>::Setup(
    KernelContext &context,
    span<const int64_t> input_shape,
    const ExtractWindowsBatchedArgs &args) {
  if (!impl || impl->IsVertical() != args.vertical) {
    impl.reset();
    if (args.vertical)
      impl = std::make_unique<ExtractVerticalWindowsImplGPU<Dst, Src>>();
    else
      impl = std::make_unique<ExtractHorizontalWindowsImplGPU<Dst, Src>>();
  }

  return impl->Setup(context, input_shape, args, args.concatenate, args.output_window_length);
}

template <typename Dst, typename Src>
void ExtractWindowsGPU<Dst, Src>::Run(
    KernelContext &context,
    const OutListGPU<Dst, 2> &output,
    const InListGPU<Src, 1> &input,
    const InTensorGPU<float, 1> &window) {
  assert(impl != nullptr);
  impl->Run(context, output, input, window);
}

template <typename Dst, typename Src>
ExtractWindowsGPU<Dst, Src>::ExtractWindowsGPU() {}

template <typename Dst, typename Src>
ExtractWindowsGPU<Dst, Src>::~ExtractWindowsGPU() {}

template class ExtractWindowsGPU<float, float>;
template class ExtractWindowsGPU<float, int16_t>;
template class ExtractWindowsGPU<float, int8_t>;

}  // namespace signal
}  // namespace kernels
}  // namespace dali
