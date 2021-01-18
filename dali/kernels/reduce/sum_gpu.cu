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

#include <memory>
#include "dali/kernels/reduce/reduce_gpu.h"
#include "dali/kernels/reduce/reduce_gpu_impl.cuh"

namespace dali {
namespace kernels {

template <typename Out, typename In>
class SumGPU<Out, In>::Impl : public reduce_impl::SumImplGPU<Out, In> {
};

template <typename Out, typename In>
SumGPU<Out, In>::SumGPU() {}

template <typename Out, typename In>
SumGPU<Out, In>::~SumGPU() {}

template <typename Out, typename In>
KernelRequirements SumGPU<Out, In>::Setup(
    KernelContext &ctx,
    const TensorListShape<> &in_shape, span<const int> axes, bool keep_dims, bool reduce_batch) {
  if (!impl_) {
    impl_ = std::make_unique<Impl>();
  }
  return impl_->Setup(ctx, in_shape, axes, keep_dims, reduce_batch);
}

template <typename Out, typename In>
void SumGPU<Out, In>::Run(KernelContext &ctx, const OutListGPU<Out> &out, const InListGPU<In> &in) {
  assert(impl_ != nullptr);
  impl_->Run(ctx, out, in);
}

template class SumGPU<uint64_t, uint8_t>;
template class SumGPU<float, uint8_t>;
template class SumGPU<int64_t, int8_t>;
template class SumGPU<float, int8_t>;
template class SumGPU<uint64_t, uint16_t>;
template class SumGPU<float, uint16_t>;
template class SumGPU<int64_t, int16_t>;
template class SumGPU<float, int16_t>;
template class SumGPU<int64_t, int32_t>;
template class SumGPU<float, int32_t>;
template class SumGPU<uint64_t, uint32_t>;
template class SumGPU<float, uint32_t>;
template class SumGPU<uint8_t, uint8_t>;
template class SumGPU<int8_t, int8_t>;
template class SumGPU<uint16_t, uint16_t>;
template class SumGPU<int16_t, int16_t>;
template class SumGPU<uint32_t, uint32_t>;
template class SumGPU<int32_t, int32_t>;
template class SumGPU<uint64_t, uint64_t>;
template class SumGPU<int64_t, int64_t>;
template class SumGPU<float, float>;

}  // namespace kernels
}  // namespace dali
