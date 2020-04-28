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

#include <vector>

// general includes
#include "dali/core/common.h"
#include "dali/pipeline/data/views.h"

// slicing includes
#include "dali/kernels/slice/slice_cpu.h"
#include "dali/core/static_switch.h"
#include "dali/operators/reader/loader/utils/readslice.h"

namespace dali {

// shape preprocessor: fuse contiguous dims
void FuseShapes(TensorShape<>& in_shape, TensorShape<>& out_shape, TensorShape<>& anchor) {
  std::vector<int64_t> in_new;
  std::vector<int64_t> out_new;
  std::vector<int64_t> anc_new;
  auto ndims = in_shape.sample_dim();
  auto in_strides = kernels::GetStrides(in_shape);
  auto in_dimsize = in_shape[ndims - 1];
  auto out_dimsize = out_shape[ndims - 1];
  auto anc_pos = anchor[ndims - 1];
  int64_t curr_stride = 1;

  for (int i = ndims - 1; i >= 1; --i) {
    if (out_shape[i] == in_shape[i]) {
      // fuse dims
      in_dimsize *= in_shape[i - 1];
      out_dimsize *= out_shape[i - 1];
      curr_stride *= in_shape[i];
      anc_pos += anchor[i - 1] * curr_stride;
    } else {
      // append the fused dim
      in_new.insert(in_new.begin(), in_dimsize);
      out_new.insert(out_new.begin(), out_dimsize);
      anc_new.insert(anc_new.begin(), anc_pos);
      in_dimsize = in_shape[i - 1];
      out_dimsize = out_shape[i - 1];
      curr_stride = 1;
      anc_pos = anchor[i - 1];
    }
  }
  // we need a final insert here
  in_new.insert(in_new.begin(), in_dimsize);
  out_new.insert(out_new.begin(), out_dimsize);
  anc_new.insert(anc_new.begin(), anc_pos);

  // reset shapes
  out_shape = TensorShape<>(out_new);
  in_shape = TensorShape<>(in_new);
  anchor = TensorShape<>(anc_new);
}

// helper function to find larger strides for better IO performance
// the stage 1 shape can be used to do the IO and the stage2 shapes
// to use the copy slice kernel
bool SplitTwostageShapes(TensorShape<>& anchor_stage1,
                         TensorShape<>& anchor_stage2,
                         TensorShape<>& out_shape_stage1,
                         const TensorShape<>& in_shape,
                         const TensorShape<>& anchor,
                         const TensorShape<>& out_shape,
                         const TypeInfo& input_type,
                         const size_t& min_read_bytes) {
  // determine whether we can use 2stage or single stage is sufficient:
  bool use_twostage = false;

  // get the strides
  auto in_strides = kernels::GetStrides(in_shape);
  auto tsize = input_type.size();

  // initialize the shapes
  out_shape_stage1 = in_shape;
  anchor_stage1 = anchor;
  for (int i = 0; i < anchor_stage1.sample_dim(); ++i)
    anchor_stage1[i] = 0;
  anchor_stage2 = anchor;

  // determine shapes
  for (int i = 0; i < in_shape.sample_dim(); ++i) {
    if ((in_strides[i] * tsize) > min_read_bytes) {
      out_shape_stage1[i] = out_shape[i];
      anchor_stage1[i] = anchor[i];
      anchor_stage2[i] = 0;
    } else {
      // from here on out, out_shape_new = in_shape
      use_twostage = true;
      break;
    }
  }
  // return the modified out_shape
  return use_twostage;
}

// this can be used to copy a slice from a file
void ReadSliceKernel(Tensor<CPUBackend>& output,
                     std::unique_ptr<FileStream>& file,
                     size_t offset,
                     const TensorShape<>& input_shape,
                     const TypeInfo& input_type,
                     const TensorShape<>& anchor,
                     const TensorShape<>& shape) {
  // resize output tensor
  output.Resize(shape, input_type);

  // fuse shapes
  TensorShape<> in_shape_new(input_shape);
  TensorShape<> out_shape_new(shape);
  TensorShape<> anchor_new(anchor);
  FuseShapes(in_shape_new, out_shape_new, anchor_new);

  // compute template switch arguments
  auto in_type = input_type.id();
  auto ndims = in_shape_new.sample_dim();

  // launch kernel
  TYPE_SWITCH(in_type, type2id, Type, READSLICE_ALLOWED_TYPES, (
    VALUE_SWITCH(ndims, Dims, READSLICE_ALLOWED_DIMS, (
      // shapes and strides
      const auto &out_shape = out_shape_new.to_static<Dims>();;
      const auto &in_shape = in_shape_new.to_static<Dims>();
      const auto &anchor_shape = anchor_new.to_static<Dims>();
      auto in_strides = kernels::GetStrides(in_shape);
      auto out_strides = kernels::GetStrides(out_shape);

      // tensor views
      auto out_tensor = make_tensor_cpu<Dims>(output.mutable_data<Type>(), out_shape);

      // tensor pointers
      Type *out_ptr = out_tensor.data;

      // launch kernel
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
  // reshape output tensor to expected shape
  auto input_type = input.type();
  output.Resize(shape, input_type);

  // do some shape preprocessing
  TensorShape<> in_shape_new(input.shape());
  TensorShape<> out_shape_new(shape);
  TensorShape<> anchor_new(anchor);
  FuseShapes(in_shape_new, out_shape_new, anchor_new);

  // compute template switch arguments
  auto in_type = input_type.id();
  auto ndims = in_shape_new.sample_dim();

  // do the slicing
  TYPE_SWITCH(in_type, type2id, Type, READSLICE_ALLOWED_TYPES, (
    VALUE_SWITCH(ndims, Dims, READSLICE_ALLOWED_DIMS, (
      // shapes and strides
      const auto &in_shape = in_shape_new.to_static<Dims>();
      const auto &out_shape = out_shape_new.to_static<Dims>();
      const auto &anchor_shape = anchor_new.to_static<Dims>();
      auto in_strides = kernels::GetStrides(in_shape);
      auto out_strides = kernels::GetStrides(out_shape);

      // tensor views
      auto out_tensor = make_tensor_cpu<Dims>(output.mutable_data<Type>(), out_shape);
      const auto in_tensor = make_tensor_cpu<Dims>(input.data<Type>(), in_shape);

      // tensor pointers
      Type *out_ptr = out_tensor.data;
      const Type *in_ptr = in_tensor.data;

      // launch kernel
      kernels::SliceKernel<Type, Type, Dims>(out_ptr,
                                             in_ptr,
                                             in_strides,
                                             out_strides,
                                             anchor_shape,
                                             out_shape);
    ), DALI_FAIL(make_string("Number of dimensions not supported ", ndims)););  // NOLINT
  ), DALI_FAIL(make_string("Type not supported", in_type)););  // NOLINT
}

}  // namespace dali
