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

#ifndef DALI_KERNELS_TEST_RESAMPLING_TEST_TRANSPOSE_OUTER_H_
#define DALI_KERNELS_TEST_RESAMPLING_TEST_TRANSPOSE_OUTER_H_

#include <utility>
#include "dali/core/tensor_view.h"
#include "dali/kernels/transpose/transpose.h"
#include "dali/kernels/transpose/transpose_gpu.h"

namespace dali {
namespace testing {

template <int ndim>
TensorShape<ndim> TransposeOuter(TensorShape<ndim> sh) {
  static_assert(ndim == DynamicDimensions || ndim >= 2,
                "Cannot transpose a tensor with fewer than 2 dimensions.");
  assert(sh.size() >= 2 && "Cannot transpose a tensor with fewer than 2 dimensions.");
  std::swap(sh[0], sh[1]);
  return sh;
}

void TransposeOuterCPU(void *out, const void *in, int64_t rows, int64_t cols, int64_t inner_size);

void TransposeOuterGPU(void *out, const void *in, int64_t rows, int64_t cols, int64_t inner_size,
                       cudaStream_t stream);

template <typename Storage, typename T, typename U, int ndim>
void TransposeOuter(const TensorView<Storage, T, ndim> &out,
                    const TensorView<Storage, U, ndim> &in, cudaStream_t stream = 0) {
  static_assert(std::is_same<std::remove_cv_t<T>, std::remove_cv_t<U>>::value,
                "The arguments may vary only by CV qualifiers");
  static_assert(std::is_trivially_copyable<T>::value,
                "The element type must be trivially copyable");
  assert(out.shape == TransposeOuter(in.shape));
  char *raw_out = reinterpret_cast<char *>(out.data);
  const char *raw_in = reinterpret_cast<char *>(in.data);
  size_t n = ndim > 2 ? volume(in.shape.last(in.dim() - 2)) * sizeof(T) : sizeof(T);
  if (std::is_same<Storage, StorageCPU>::value)
    TransposeOuterCPU(raw_out, raw_in, in.shape[0], in.shape[1], n);
  else
    TransposeOuterGPU(raw_out, raw_in, in.shape[0], in.shape[1], n, stream);
}

}  // namespace testing
}  // namespace dali

#endif  // DALI_KERNELS_TEST_RESAMPLING_TEST_TRANSPOSE_OUTER_H_
