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


// general includes
#include "dali/core/common.h"
#include "dali/pipeline/data/views.h"

// slicing includes
#include "dali/kernels/slice/slice_cpu.h"
#include "dali/core/static_switch.h"
#include "dali/operators/reader/loader/utils/readslice.h"

namespace dali {

// this can be used to copy a slice from a file
void ReadSliceKernel(Tensor<CPUBackend>& output,
                     std::unique_ptr<FileStream>& file,
                     size_t offset,
                     const TensorShape<>& input_shape,
                     const TypeInfo& input_type,
                     const TensorShape<>& anchor,
                     const TensorShape<>& shape) {
  // do the slicing
  output.Resize(shape, input_type);
  auto ndims = input_shape.sample_dim();
  auto in_type = input_type.id();
  TYPE_SWITCH(in_type, type2id, Type, READSLICE_ALLOWED_TYPES, (
    VALUE_SWITCH(ndims, Dims, READSLICE_ALLOWED_DIMS, (
      auto out_tensor = view<Type, Dims>(output);
      Type *out_ptr = out_tensor.data;
      const auto &out_shape = out_tensor.shape;
      const auto &in_shape = input_shape.to_static<Dims>();
      const auto &anchor_shape = anchor.to_static<Dims>();
      auto in_strides = kernels::GetStrides(in_shape);
      auto out_strides = kernels::GetStrides(out_shape);
      ReadSliceKernel<FileStream, Type, Dims>(out_ptr,
                                              file,
                                              offset,
                                              in_strides,
                                              out_strides,
                                              anchor_shape,
                                              out_shape);
    ), DALI_FAIL(make_string("Number of dimensions not supported ", ndims)););  // NOLINT
  ), DALI_FAIL(make_string("Type not supported", in_type)););  // NOLINT   
}

// this can be used to copy a slice from a memmapped buffer
void CopySliceKernel(Tensor<CPUBackend>& output,
                     const Tensor<CPUBackend>& input,
                     const TensorShape<>& anchor,
                     const TensorShape<>& shape) {
  // do the slicing
  output.Resize(shape, input.type());
  auto ndims = input.shape().sample_dim();
  auto input_type = input.type().id();
  TYPE_SWITCH(input_type, type2id, Type, READSLICE_ALLOWED_TYPES, (
    VALUE_SWITCH(ndims, Dims, READSLICE_ALLOWED_DIMS, (
      auto out_tensor = view<Type, Dims>(output);
      const auto& in_tensor = view<const Type, Dims>(input);
      Type *out_ptr = out_tensor.data;
      const Type *in_ptr = in_tensor.data;
      const auto &in_shape = in_tensor.shape;
      const auto &out_shape = out_tensor.shape;
      const auto &anchor_shape = anchor.to_static<Dims>();
      auto in_strides = kernels::GetStrides(in_shape);
      auto out_strides = kernels::GetStrides(out_shape);
      kernels::SliceKernel<Type, Type, Dims>(out_ptr,
                                             in_ptr,
                                             in_strides,
                                             out_strides,
                                             anchor_shape,
                                             out_shape);
    ), DALI_FAIL(make_string("Number of dimensions not supported ", ndims)););  // NOLINT
  ), DALI_FAIL(make_string("Type not supported", input_type)););  // NOLINT
}

}  // namespace dali
