// Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_TEST_TEST_TENSORS_H_
#define DALI_TEST_TEST_TENSORS_H_

#include <cuda_runtime_api.h>
#include <memory>
#include <utility>
#include "dali/core/cuda_error.h"
#include "dali/core/tensor_view.h"
#include "dali/core/backend_tags.h"
#include "dali/core/mm/memory.h"

namespace dali {
namespace kernels {

template <typename T, int dim = DynamicDimensions>
class TestTensorList {
 public:
  void reshape(const TensorListShape<dim> &shape) {
    cpumem_.reset();
    gpumem_.reset();
    this->shape_ = shape;
  }
  void reshape(TensorListShape<dim> &&shape) {
    cpumem_.reset();
    gpumem_.reset();
    this->shape_ = std::move(shape);
  }

  void invalidate_cpu() {
    cpumem_.reset();
  }

  void invalidate_gpu() {
    gpumem_.reset();
  }

  template <int out_dim = dim>
  TensorListView<StorageCPU, T, out_dim> cpu(cudaStream_t stream = 0) {
    TensorListView<StorageCPU, T, out_dim> ret;
    if (!cpumem_) {
      auto size = shape_.num_elements() * sizeof(T);
      cpumem_ = mm::alloc_raw_unique<char, mm::memory_kind::host>(size, 256);
      if (gpumem_)
        CUDA_CALL(cudaMemcpyAsync(cpumem_.get(), gpumem_.get(), size,
                                  cudaMemcpyDeviceToHost, stream));
    }
    auto out_shape = convert_dim<out_dim>(shape_);
    return { reinterpret_cast<T*>(cpumem_.get()), std::move(out_shape) };
  }

  template <int out_dim = dim>
  TensorListView<StorageGPU, T, out_dim> gpu(cudaStream_t stream = 0) {
    TensorListView<StorageGPU, T, out_dim> ret;
    if (!gpumem_) {
      auto size = shape_.num_elements() * sizeof(T);
      gpumem_ = mm::alloc_raw_unique<char, mm::memory_kind::device>(size, 256);
      if (cpumem_)
        CUDA_CALL(cudaMemcpyAsync(gpumem_.get(), cpumem_.get(), size,
                                  cudaMemcpyHostToDevice, stream));
    }
    auto out_shape = convert_dim<out_dim>(shape_);
    return { reinterpret_cast<T*>(gpumem_.get()), std::move(out_shape) };
  }

  // workaround for lack of partial specialization for functions
  template <int out_dim = dim>
  TensorListView<StorageGPU, T, out_dim> get(StorageGPU, cudaStream_t stream = 0) {
    return gpu<out_dim>(stream);
  }

  // workaround for lack of partial specialization for functions
  template <int out_dim = dim>
  TensorListView<StorageCPU, T, out_dim> get(StorageCPU, cudaStream_t stream = 0) {
    return cpu<out_dim>(stream);
  }

  template <typename Backend, int out_dim = dim>
  TensorListView<Backend, T, out_dim> get(cudaStream_t stream = 0) {
    return get<out_dim>(Backend(), stream);
  }

 private:
  mm::uptr<char> cpumem_, gpumem_;
  TensorListShape<dim> shape_;
};


}  // namespace kernels
}  // namespace dali

#endif  // DALI_TEST_TEST_TENSORS_H_
