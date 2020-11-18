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
#include "dali/kernels/reduce/mean_stddev_gpu_impl.cuh"

namespace dali {
namespace kernels {

template <typename Out, typename In>
class MeanGPU<Out, In>::Impl : public reduce_impl::MeanImplGPU<Out, In> {
};

template <typename Out, typename In>
MeanGPU<Out, In>::MeanGPU() {}

template <typename Out, typename In>
MeanGPU<Out, In>::~MeanGPU() {}

template <typename Out, typename In>
KernelRequirements MeanGPU<Out, In>::Setup(
    KernelContext &ctx,
    const TensorListShape<> &in_shape, span<const int> axes, bool keep_dims, bool reduce_batch) {
  if (!impl_) {
    impl_ = std::make_unique<Impl>();
  }
  return impl_->Setup(ctx, in_shape, axes, keep_dims, reduce_batch);
}

template <typename Out, typename In>
void MeanGPU<Out, In>::Run(
    KernelContext &ctx, const OutListGPU<Out> &out, const InListGPU<In> &in) {
  assert(impl_ != nullptr);
  impl_->Run(ctx, out, in);
}

template class MeanGPU<uint8_t, uint8_t>;
template class MeanGPU<float, uint8_t>;
template class MeanGPU<int8_t, int8_t>;
template class MeanGPU<float, int8_t>;
template class MeanGPU<uint16_t, uint16_t>;
template class MeanGPU<float, uint16_t>;
template class MeanGPU<int16_t, int16_t>;
template class MeanGPU<float, int16_t>;
template class MeanGPU<int32_t, int32_t>;
template class MeanGPU<float, int32_t>;
template class MeanGPU<uint32_t, uint32_t>;
template class MeanGPU<float, uint32_t>;
template class MeanGPU<int64_t, int64_t>;
template class MeanGPU<float, int64_t>;
template class MeanGPU<uint64_t, uint64_t>;
template class MeanGPU<float, uint64_t>;
template class MeanGPU<float, float>;


template <typename Out, typename In>
class MeanSquareGPU<Out, In>::Impl : public reduce_impl::MeanSquareImplGPU<Out, In> {
};

template <typename Out, typename In>
MeanSquareGPU<Out, In>::MeanSquareGPU() {}

template <typename Out, typename In>
MeanSquareGPU<Out, In>::~MeanSquareGPU() {}

template <typename Out, typename In>
KernelRequirements MeanSquareGPU<Out, In>::Setup(
    KernelContext &ctx,
    const TensorListShape<> &in_shape, span<const int> axes, bool keep_dims, bool reduce_batch) {
  if (!impl_) {
    impl_ = std::make_unique<Impl>();
  }
  return impl_->Setup(ctx, in_shape, axes, keep_dims, reduce_batch);
}

template <typename Out, typename In>
void MeanSquareGPU<Out, In>::Run(
    KernelContext &ctx, const OutListGPU<Out> &out, const InListGPU<In> &in) {
  assert(impl_ != nullptr);
  impl_->Run(ctx, out, in);
}

template class MeanSquareGPU<uint64_t, uint8_t>;
template class MeanSquareGPU<float, uint8_t>;
template class MeanSquareGPU<int64_t, int8_t>;
template class MeanSquareGPU<float, int8_t>;
template class MeanSquareGPU<uint64_t, uint16_t>;
template class MeanSquareGPU<float, uint16_t>;
template class MeanSquareGPU<int64_t, int16_t>;
template class MeanSquareGPU<float, int16_t>;
template class MeanSquareGPU<int64_t, int32_t>;
template class MeanSquareGPU<float, int32_t>;
template class MeanSquareGPU<uint64_t, uint32_t>;
template class MeanSquareGPU<float, uint32_t>;
template class MeanSquareGPU<float, float>;
template class MeanSquareGPU<uint64_t, uint64_t>;
template class MeanSquareGPU<float, uint64_t>;
template class MeanSquareGPU<int64_t, int64_t>;
template class MeanSquareGPU<float, int64_t>;


template <typename Out, typename In>
class RootMeanSquareGPU<Out, In>::Impl : public reduce_impl::RootMeanSquareImplGPU<Out, In> {
};

template <typename Out, typename In>
RootMeanSquareGPU<Out, In>::RootMeanSquareGPU() {}

template <typename Out, typename In>
RootMeanSquareGPU<Out, In>::~RootMeanSquareGPU() {}

template <typename Out, typename In>
KernelRequirements RootMeanSquareGPU<Out, In>::Setup(
    KernelContext &ctx,
    const TensorListShape<> &in_shape, span<const int> axes, bool keep_dims, bool reduce_batch) {
  if (!impl_) {
    impl_ = std::make_unique<Impl>();
  }
  return impl_->Setup(ctx, in_shape, axes, keep_dims, reduce_batch);
}

template <typename Out, typename In>
void RootMeanSquareGPU<Out, In>::Run(
    KernelContext &ctx, const OutListGPU<Out> &out, const InListGPU<In> &in) {
  assert(impl_ != nullptr);
  impl_->Run(ctx, out, in);
}

template class RootMeanSquareGPU<uint8_t, uint8_t>;
template class RootMeanSquareGPU<float, uint8_t>;
template class RootMeanSquareGPU<int8_t, int8_t>;
template class RootMeanSquareGPU<float, int8_t>;
template class RootMeanSquareGPU<uint16_t, uint16_t>;
template class RootMeanSquareGPU<float, uint16_t>;
template class RootMeanSquareGPU<int16_t, int16_t>;
template class RootMeanSquareGPU<float, int16_t>;
template class RootMeanSquareGPU<int32_t, int32_t>;
template class RootMeanSquareGPU<float, int32_t>;
template class RootMeanSquareGPU<uint32_t, uint32_t>;
template class RootMeanSquareGPU<float, uint32_t>;
template class RootMeanSquareGPU<float, float>;
template class RootMeanSquareGPU<uint64_t, uint64_t>;
template class RootMeanSquareGPU<float, uint64_t>;
template class RootMeanSquareGPU<int64_t, int64_t>;
template class RootMeanSquareGPU<float, int64_t>;

}  // namespace kernels
}  // namespace dali
