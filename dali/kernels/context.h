// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_CONTEXT_H_
#define DALI_KERNELS_CONTEXT_H_

#include <cuda_runtime_api.h>
#include <vector>
#include "dali/kernels/tensor_view.h"
#include "dali/kernels/alloc_type.h"

namespace dali {
namespace kernels {

template <typename ComputeBackend>
struct Context {};

template <>
struct Context<ComputeGPU> {
  cudaStream_t stream = 0;
};

class ScratchpadAllocator {
 public:
  virtual void *Alloc(AllocType alloc_type, size_t bytes, size_t alignment) = 0;

  template <AllocType alloc_type, typename T, size_t dim>
  TensorView<AllocBackend<alloc_type>, T, dim> AllocTensor(TensorShape<dim> shape) {
    return { New<T>(alloc_type, Volume(shape)), shape };
  }

  template <AllocType alloc_type, typename T, size_t dim>
  TensorListView<AllocBackend<alloc_type>, T, dim>
  AllocTensorList(std::vector<TensorShape<dim>> shape) {
    TensorListView<AllocBackend<alloc_type>, T, dim> tlv(nullptr, shape);
    tlv.data = New<T>(alloc_type, tlv.total_size());
    return tlv;
  }

  template <typename T>
  T *New(AllocType alloc_type, size_t count, size_t alignment = alignof(T)) {
    return reinterpret_cast<T*>(Alloc(alloc_type, count*sizeof(T), alignment));
  }

 protected:
  ~ScratchpadAllocator() = default;
};

using CPUContext = Context<ComputeCPU>;
using GPUContext = Context<ComputeGPU>;

struct KernelContext {
  CPUContext cpu;
  GPUContext gpu;
  ScratchpadAllocator *scratchpad;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_CONTEXT_H_
