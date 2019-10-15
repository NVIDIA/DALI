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
#include <utility>
#include "dali/core/traits.h"
#include "dali/kernels/alloc.h"
#include "dali/core/backend_tags.h"
#include "dali/core/tensor_view.h"

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


/**
 * Copies input TensorView and returns the output.
 * @tparam DstAlloc Requested allocation type of the output TensorView.
 *                  According to this parameter, StorageBackend of output TensorView will be determined
 * @tparam NonconstT utility parameter, do not specify (leave default)
 * @return The output consists of new TensorView along with pointer to its memory (as the TensorView doesn't own any)
 */
template<AllocType DstAlloc, typename SrcBackend, typename T, int ndims>
std::pair<
        TensorView<AllocBackend<DstAlloc>, dali::remove_const_t<T>, ndims>,
        memory::KernelUniquePtr<dali::remove_const_t<T>>
          >
copy(const TensorView <SrcBackend, T, ndims> &src) {
  auto mem = kernels::memory::alloc_unique<dali::remove_const_t<T>>(DstAlloc, volume(src.shape));
  auto tvgpu = make_tensor<AllocBackend<DstAlloc>, ndims>(mem.get(), src.shape);
  kernels::copy(tvgpu, src);
  return std::make_pair(tvgpu, std::move(mem));
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_COPY_H_
