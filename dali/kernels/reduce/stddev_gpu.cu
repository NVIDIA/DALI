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

template <typename Out, typename In, typename Mean>
class StdDevGPU<Out, In, Mean>::Impl : public reduce_impl::StdDevImplGPU<Out, In, Mean> {
};

template <typename Out, typename In, typename Mean>
StdDevGPU<Out, In, Mean>::StdDevGPU() {}

template <typename Out, typename In, typename Mean>
StdDevGPU<Out, In, Mean>::~StdDevGPU() {}

template <typename Out, typename In, typename Mean>
KernelRequirements StdDevGPU<Out, In, Mean>::Setup(
    KernelContext &ctx,
    const TensorListShape<> &in_shape, span<const int> axes, bool keep_dims, bool reduce_batch) {
  if (!impl_) {
    impl_ = std::make_unique<Impl>();
  }
  return impl_->Setup(ctx, in_shape, axes, keep_dims, reduce_batch);
}

template <typename Out, typename In, typename Mean>
void StdDevGPU<Out, In, Mean>::Run(KernelContext &ctx, const OutListGPU<Out> &out,
                             const InListGPU<In> &in, const InListGPU<Mean> &mean,
                             int ddof) {
  assert(impl_ != nullptr);
  impl_->Run(ctx, out, in, mean, ddof);
}


template class StdDevGPU<uint8_t, uint8_t>;
template class StdDevGPU<float, uint8_t>;
template class StdDevGPU<int8_t, int8_t>;
template class StdDevGPU<float, int8_t>;

template class StdDevGPU<uint16_t, uint16_t>;
template class StdDevGPU<float, uint16_t>;
template class StdDevGPU<int16_t, int16_t>;
template class StdDevGPU<float, int16_t>;

template class StdDevGPU<uint32_t, uint32_t>;
template class StdDevGPU<float, uint32_t>;
template class StdDevGPU<int32_t, int32_t>;
template class StdDevGPU<float, int32_t>;

template class StdDevGPU<uint64_t, uint64_t>;
template class StdDevGPU<float, uint64_t>;
template class StdDevGPU<int64_t, int64_t>;
template class StdDevGPU<float, int64_t>;

template class StdDevGPU<float, float>;


template <typename Out, typename In, typename Mean>
class VarianceGPU<Out, In, Mean>::Impl : public reduce_impl::VarianceImplGPU<Out, In, Mean> {
};

template <typename Out, typename In, typename Mean>
VarianceGPU<Out, In, Mean>::VarianceGPU() {}

template <typename Out, typename In, typename Mean>
VarianceGPU<Out, In, Mean>::~VarianceGPU() {}

template <typename Out, typename In, typename Mean>
KernelRequirements VarianceGPU<Out, In, Mean>::Setup(
    KernelContext &ctx,
    const TensorListShape<> &in_shape, span<const int> axes, bool keep_dims, bool reduce_batch) {
  if (!impl_) {
    impl_ = std::make_unique<Impl>();
  }
  return impl_->Setup(ctx, in_shape, axes, keep_dims, reduce_batch);
}

template <typename Out, typename In, typename Mean>
void VarianceGPU<Out, In, Mean>::Run(KernelContext &ctx, const OutListGPU<Out> &out,
                             const InListGPU<In> &in, const InListGPU<Mean> &mean,
                             int ddof) {
  assert(impl_ != nullptr);
  impl_->Run(ctx, out, in, mean, ddof);
}


template class VarianceGPU<uint8_t, uint8_t>;
template class VarianceGPU<float, uint8_t>;
template class VarianceGPU<int8_t, int8_t>;
template class VarianceGPU<float, int8_t>;

template class VarianceGPU<uint16_t, uint16_t>;
template class VarianceGPU<float, uint16_t>;
template class VarianceGPU<int16_t, int16_t>;
template class VarianceGPU<float, int16_t>;

template class VarianceGPU<uint32_t, uint32_t>;
template class VarianceGPU<float, uint32_t>;
template class VarianceGPU<int32_t, int32_t>;
template class VarianceGPU<float, int32_t>;

template class VarianceGPU<uint64_t, uint64_t>;
template class VarianceGPU<float, uint64_t>;
template class VarianceGPU<int64_t, int64_t>;
template class VarianceGPU<float, int64_t>;

template class VarianceGPU<float, float>;


template <typename Out, typename In, typename Mean>
class InvStdDevGPU<Out, In, Mean>::Impl : public reduce_impl::InvStdDevImplGPU<Out, In, Mean> {
};

template <typename Out, typename In, typename Mean>
InvStdDevGPU<Out, In, Mean>::InvStdDevGPU() {}

template <typename Out, typename In, typename Mean>
InvStdDevGPU<Out, In, Mean>::~InvStdDevGPU() {}

template <typename Out, typename In, typename Mean>
KernelRequirements InvStdDevGPU<Out, In, Mean>::Setup(
    KernelContext &ctx,
    const TensorListShape<> &in_shape, span<const int> axes, bool keep_dims, bool reduce_batch) {
  if (!impl_) {
    impl_ = std::make_unique<Impl>();
  }
  return impl_->Setup(ctx, in_shape, axes, keep_dims, reduce_batch);
}

template <typename Out, typename In, typename Mean>
void InvStdDevGPU<Out, In, Mean>::Run(
    KernelContext &ctx, const OutListGPU<Out> &out,
    const InListGPU<In> &in, const InListGPU<Mean> &mean, int ddof, param_t epsilon) {
  assert(impl_ != nullptr);
  impl_->Run(ctx, out, in, mean, ddof, epsilon);
}

template class InvStdDevGPU<float, uint8_t>;
template class InvStdDevGPU<float, int8_t>;

template class InvStdDevGPU<float, uint16_t>;
template class InvStdDevGPU<float, int16_t>;

template class InvStdDevGPU<float, uint32_t>;
template class InvStdDevGPU<float, int32_t>;

template class InvStdDevGPU<float, float>;

}  // namespace kernels
}  // namespace dali
