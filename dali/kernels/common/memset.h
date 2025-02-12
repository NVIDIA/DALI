// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_COMMON_MEMSET_H_
#define DALI_KERNELS_COMMON_MEMSET_H_

#include <cuda_runtime.h>
#include <cstring>
#include <utility>
#include "dali/core/backend_tags.h"
#include "dali/core/cuda_error.h"
#include "dali/core/mm/memory.h"
#include "dali/core/tensor_view.h"
#include "dali/core/traits.h"

namespace dali {
namespace kernels {

template <typename Storage>
void memset(void *out, uint8_t value, std::size_t N, cudaStream_t stream = 0) {
  if (is_cpu_accessible<Storage>::value) {
    std::memset(out, value, N);
  } else {
    CUDA_CALL(cudaMemsetAsync(out, value, N, stream));
  }
}

template <typename Storage, typename T, int NDim>
void memset(const TensorView<Storage, T, NDim> &out, int value, cudaStream_t stream = 0) {
  static_assert(!std::is_const<T>::value, "Cannot memset a tensor of const elements!");
  memset<Storage>(out.data, value, out.num_elements() * sizeof(T), stream);
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_MEMSET_H_
