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

#ifndef DALI_OPERATORS_READER_LOADER_UTILS_READSLICE_H_
#define DALI_OPERATORS_READER_LOADER_UTILS_READSLICE_H_

#include <memory>

#include "dali/core/common.h"
#include "dali/pipeline/data/types.h"
#include "dali/util/file.h"

namespace dali {

// some defines
#define READSLICE_ALLOWED_DIMS (1, 2, 3, 4, 5)

#define READSLICE_ALLOWED_TYPES \
  (bool, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, float16, \
  double)

namespace detail {

template <class Fstream, typename Type, int Dims>
void ReadSliceKernelImpl(Type *output,
                         std::unique_ptr<Fstream>& file,
                         size_t offset,
                         const TensorShape<Dims> &in_strides,
                         const TensorShape<Dims> &out_strides,
                         const TensorShape<Dims> &out_shape,
                         std::integral_constant<int, 1>) {
  file->Seek(offset);
  file->Read(reinterpret_cast<uint8_t*>(output), out_shape[Dims - 1] * sizeof(Type));
}

template <class Fstream, typename Type, int Dims, int DimsLeft>
void ReadSliceKernelImpl(Type *output,
                         std::unique_ptr<Fstream>& file,
                         size_t offset,
                         const TensorShape<Dims> &in_strides,
                         const TensorShape<Dims> &out_strides,
                         const TensorShape<Dims> &out_shape,
                         std::integral_constant<int, DimsLeft>) {
  constexpr auto d = Dims - DimsLeft;  // NOLINT
  for (int i = 0; i < out_shape[d]; i++) {
    ReadSliceKernelImpl(output, file, offset,
                        in_strides, out_strides, out_shape,
                        std::integral_constant<int, DimsLeft - 1>());
    offset += in_strides[d] * sizeof(Type);
    output += out_strides[d];
  }
}

}  // namespace detail

template <class Fstream, typename Type, int Dims>
void ReadSliceKernel(Type *output,
                     std::unique_ptr<Fstream>& file,
                     size_t offset,
                     const TensorShape<Dims> &in_strides,
                     const TensorShape<Dims> &out_strides,
                     const TensorShape<Dims> &anchor,
                     const TensorShape<Dims> &out_shape) {
  for (int d = 0; d < Dims; d++) {
    offset += in_strides[d] * anchor[d] * sizeof(Type);
  }
  detail::ReadSliceKernelImpl(output, file, offset,
                              in_strides, out_strides, out_shape,
                              std::integral_constant<int, Dims>());
}

// we need to use this when we use the read-based file IO
void ReadSliceKernel(Tensor<CPUBackend>& output,
                     std::unique_ptr<FileStream>& file,
                     size_t offset,
                     const TensorShape<>& input_shape,
                     const TypeInfo& input_type,
                     const TensorShape<>& anchor,
                     const TensorShape<>& shape);

// we need to use this when we use the memmapped-based file IO
void CopySliceKernel(Tensor<CPUBackend>& output,
                     const Tensor<CPUBackend>& input,
                     const TensorShape<>& anchor,
                     const TensorShape<>& shape);

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_UTILS_READSLICE_H_
