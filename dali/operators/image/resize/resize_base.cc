// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/imgproc/resample.h"
#include "dali/kernels/imgproc/resample_cpu.h"
#include "dali/operators/image/resize/resize_base.h"
#include "dali/pipeline/data/views.h"
#include "dali/core/static_switch.h"

#define DALI_RESIZE_BASE_CC 1
#include "dali/operators/image/resize/resize_op_impl_cpu.h"
#include "dali/operators/image/resize/resize_op_impl_gpu.h"

namespace dali {

template <typename Backend>
ResizeBase<Backend>::ResizeBase(const OpSpec &spec) {
  size_t temp_buffer_hint = spec.GetArgument<int64_t>("temp_buffer_hint");
}

template <typename Backend>
void ResizeBase<Backend>::SetupResize(TensorListShape<> &out_shape,
                                      DALIDataType out_type,
                                      const TensorListShape<> &in_shape,
                                      DALIDataType in_type,
                                      span<const kernels::ResamplingParams> params,
                                      int spatial_ndim,
                                      int first_spatial_dim) {
  if (out_type == in_type) {
    TYPE_SWITCH(out_type, type2id, OutputType, (uint8_t, int16_t, uint16_t, float),
      (this->template SetupResizeTyped<OutputType, OutputType>(out_shape, in_shape, params,
        spatial_ndim, first_spatial_dim)),
      (DALI_FAIL(make_string("Unsupported type: ", out_type,
        ". Supported types are: uint8, int16, uint16 and float"))));
  } else {
    DALI_ENFORCE(out_type == DALI_FLOAT,
      make_string("Resize must output original type or float. Got: ", out_type));
    TYPE_SWITCH(in_type, type2id, InputType, (uint8_t, int16_t, uint16_t),
      (this->template SetupResizeTyped<float, InputType>(out_shape, in_shape, params,
        spatial_ndim, first_spatial_dim)),
      (DALI_FAIL(make_string("Unsupported type: ", in_type,
        ". Supported types are: uint8, int16, uint16 and float"))));
  }
}

template <typename Backend>
template <typename OutputType, typename InputType>
void ResizeBase<Backend>::SetupResizeTyped(
      TensorListShape<> &out_shape,
      const TensorListShape<> &in_shape,
      span<const kernels::ResamplingParams> params,
      int spatial_ndim,
      int first_spatial_dim) {
  VALUE_SWITCH(spatial_ndim, static_spatial_ndim, (2, 3),
  (SetupResizeStatic<OutputType, InputType, static_spatial_ndim>(
      out_shape, in_shape, params, first_spatial_dim)),
  (DALI_FAIL(make_string("Unsupported number of resized dimensions: ", spatial_ndim))));
}

template <>
template <typename OutputType, typename InputType, int spatial_ndim>
void ResizeBase<GPUBackend>::SetupResizeStatic(
      TensorListShape<> &out_shape,
      const TensorListShape<> &in_shape,
      span<const kernels::ResamplingParams> params,
      int first_spatial_dim) {
  using ImplType = ResizeOpImplGPU<OutputType, InputType, spatial_ndim>;
  auto *impl = dynamic_cast<ImplType*>(impl_.get());
  if (!impl) {
    impl_.reset();
    auto unq_impl = std::make_unique<ImplType>(kmgr_, minibatch_size_);
    impl = unq_impl.get();
    impl_ = std::move(unq_impl);
  }
  impl->Setup(out_shape, in_shape, first_spatial_dim, params);
}


template <>
template <typename OutputType, typename InputType, int spatial_ndim>
void ResizeBase<CPUBackend>::SetupResizeStatic(
      TensorListShape<> &out_shape,
      const TensorListShape<> &in_shape,
      span<const kernels::ResamplingParams> params,
      int first_spatial_dim) {
  using ImplType = ResizeOpImplCPU<OutputType, InputType, spatial_ndim>;
  auto *impl = dynamic_cast<ImplType*>(impl_.get());
  if (!impl) {
    impl_.reset();
    auto unq_impl = std::make_unique<ImplType>(kmgr_, num_threads_);
    impl = unq_impl.get();
    impl_ = std::move(unq_impl);
  }
  impl->Setup(out_shape, in_shape, first_spatial_dim, params);
}


template <>
void ResizeBase<CPUBackend>::InitializeCPU(int num_threads) {
  if (num_threads != num_threads_) {
    impl_.reset();
    num_threads_ = num_threads;
  }
}

template <>
void ResizeBase<GPUBackend>::InitializeGPU(int minibatch_size, size_t temp_buffer_hint) {
  if (minibatch_size != minibatch_size_) {
    impl_.reset();
    minibatch_size_ = minibatch_size;
  }
  kmgr_.Resize(1, 0);
  kmgr_.SetMemoryHint<mm::memory_kind::device>(temp_buffer_hint);
  kmgr_.GetScratchpadAllocator(0).Reserve<mm::memory_kind::device>(temp_buffer_hint);
}

template <typename Backend>
void ResizeBase<Backend>::RunResize(Workspace &ws,
                                    OutputBufferType &output,
                                    const InputBufferType &input) {
  impl_->RunResize(ws, output, input);
}


template class ResizeBase<CPUBackend>;
template class ResizeBase<GPUBackend>;

}  // namespace dali
