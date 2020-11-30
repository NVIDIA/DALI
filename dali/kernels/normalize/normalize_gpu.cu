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

#include "dali/kernels/normalize/normalize_gpu.h"
#include <memory>
#include "dali/kernels/normalize/normalize_gpu_impl.cuh"

namespace dali {
namespace kernels {

template <typename Out, typename In>
struct NormalizeGPU<Out, In>::Impl : normalize_impl::NormalizeImplGPU<Out, In, Base, Scale> {};

template <typename Out, typename In>
NormalizeGPU<Out, In>::NormalizeGPU() {}

template <typename Out, typename In>
NormalizeGPU<Out, In>::~NormalizeGPU() {}

template <typename Out, typename In>
KernelRequirements NormalizeGPU<Out, In>::Setup(
      KernelContext &ctx,
      const TensorListShape<> &data_shape,
      const TensorListShape<> &param_shape,
      bool scalar_base,
      bool scalar_scale,
      bool scale_is_stddev) {
  if (!impl_)
    impl_ = std::make_unique<Impl>();
  return impl_->Setup(ctx, data_shape, param_shape, scalar_base, scalar_scale, scale_is_stddev);
}

template <typename Out, typename In>
KernelRequirements NormalizeGPU<Out, In>::Setup(
      KernelContext &ctx,
      const TensorListShape<> &data_shape,
      span<const int> axes,
      bool scalar_base,
      bool scalar_scale,
      bool scale_is_stddev) {
  if (!impl_)
    impl_ = std::make_unique<Impl>();
  return impl_->Setup(ctx, data_shape, axes, scalar_base, scalar_scale, scale_is_stddev);
}

template <typename Out, typename In>
void NormalizeGPU<Out, In>::Run(
      KernelContext &ctx,
      const OutListGPU<Out> &out, const InListGPU<In> &in,
      const InListGPU<Base> &base, const InListGPU<Scale> &scale,
      float global_scale, float shift, float epsilon) {
  assert(impl_ && "Call setup first");
  impl_->Run(ctx, out, in, base, scale, global_scale, shift, epsilon);
}

template <typename Out, typename In>
void NormalizeGPU<Out, In>::Run(
      KernelContext &ctx,
      const OutListGPU<Out> &out, const InListGPU<In> &in,
      float base, const InListGPU<Scale> &scale,
      float global_scale, float shift, float epsilon) {
  assert(impl_ && "Call setup first");
  impl_->Run(ctx, out, in, base, scale, global_scale, shift, epsilon);
}

template <typename Out, typename In>
void NormalizeGPU<Out, In>::Run(
      KernelContext &ctx,
      const OutListGPU<Out> &out, const InListGPU<In> &in,
      const InListGPU<Base> &base, float scale,
      float global_scale, float shift, float epsilon) {
  assert(impl_ && "Call setup first");
  impl_->Run(ctx, out, in, base, scale, global_scale, shift, epsilon);
}

template <typename Out, typename In>
void NormalizeGPU<Out, In>::Run(
      KernelContext &ctx,
      const OutListGPU<Out> &out, const InListGPU<In> &in,
      float base, float scale,
      float global_scale, float shift, float epsilon) {
  assert(impl_ && "Call setup first");
  impl_->Run(ctx, out, in, base, scale, global_scale, shift, epsilon);
}

// instantiate explicitly

DALI_INSTANTIATE_NORMALIZE_GPU()

}  // namespace kernels
}  // namespace dali
