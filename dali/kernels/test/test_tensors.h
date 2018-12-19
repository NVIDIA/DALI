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

#ifndef DALI_KERNELS_TEST_TEST_TENSORS_H_
#define DALI_KERNELS_TEST_TEST_TENSORS_H_

#include <cuda_runtime_api.h>
#include <dali/kernels/tensor_view.h>

namespace dali {
namespace kernels {

template <typename T, int dim = -1>
class TestTensorList {
 public:

  template <typename T>
  TensorListView<StorageBackend::CPU, T, dim> cpu(cudaStream_t stream = 0) {
    TensorListView<StorageBackend::CPU, T, dim> ret;
    ret.set_sample_dim(shape_.sample_dim);
    ret.offsets = shape_.offsets;
    ret.data = cpu_.get();
    return ret;
  }

  template <typename T>
  TensorListView<StorageBackend::GPU, T, dim> gpu(cudaStream_t stream = 0) {
    TensorListView<StorageBackend::GPU, T, dim> ret;
    ret.set_sample_dim(shape_.sample_dim);
    ret.offsets = shape_.offsets;
    if (!gpu && cpu) {
      void *ptr;
      auto size = shape_.total_size();
      cudaMalloc(&ptr, size);
      gpu.reset(reinterpret_cast<char*>(ptr), GPUDeleter);
      cudaMemcpy(ptr, cpu.get(), size, cudaMemcpyHostToDevice);
    }
    ret.data = gpu_.get();
    return ret;
  }

private:
  // TODO(michalz): change to TensorListShape when ready
  TensorListView<StorageBackend::CPU, T, dim> shape_;
  std::unique_ptr<char, Deleter> cpu_, gpu_;

  using Deleter = void(*)(char *);
  static void CPUDeleter(char *mem) {
    delete [] mem;
  }
  static void GPUDeleter(char *mem) {
    cudaFree(mem);
  }
};


}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_TEST_TEST_TENSORS_H_
