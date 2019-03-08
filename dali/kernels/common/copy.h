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

#ifndef DALI_KERNELS_COMMON_COPY_H_
#define DALI_KERNELS_COMMON_COPY_H_

#include <cuda_runtime.h>
#include <cstring>
#include "dali/kernels/backend_tags.h"
#include "dali/kernels/tensor_view.h"

namespace dali {
namespace kernels {

template <typename StorageOut, typename StorageIn>
void copy(void* out, const void* in, std::size_t N, cudaStream_t stream = 0) {
  if (!is_gpu_accessible<StorageOut>::value) {
    if (is_cpu_accessible<StorageIn>::value) {
      if (is_gpu_accessible<StorageIn>::value)
        cudaStreamSynchronize(stream);  // or cudaDeviceSynchronize?
      std::memcpy(out, in, N);
    } else {
      cudaMemcpyAsync(out, in, N, cudaMemcpyDeviceToHost, stream);
    }
  } else {
    if (is_gpu_accessible<StorageIn>::value) {
      cudaMemcpyAsync(out, in, N, cudaMemcpyDeviceToDevice, stream);
    } else {
      cudaMemcpyAsync(out, in, N, cudaMemcpyHostToDevice, stream);
    }
  }
}

template <typename StorageOut, typename TOut, int NDimOut,
          typename StorageIn, typename TIn, int NDimIn>
void copy(const TensorView<StorageOut, TOut, NDimIn>& out,
          const TensorView<StorageIn, TIn, NDimOut>& in, cudaStream_t stream = 0) {
  static_assert(sizeof(TOut) == sizeof(TIn), "Tensor elements must be of equal size!");
  static_assert(!std::is_const<TOut>::value, "Cannot copy to a tensor of const elements!");
  assert(in.shape == out.shape);
  copy<StorageOut, StorageIn>(out.data, in.data, in.num_elements() * sizeof(TOut), stream);
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_COPY_H_
