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

#ifndef DALI_CORE_DEV_BUFFER_H_
#define DALI_CORE_DEV_BUFFER_H_

#include <cuda_runtime_api.h>
#include <functional>
#include <memory>
#include <utility>
#include "dali/core/cuda_error.h"
#include "dali/core/util.h"

namespace dali {

template <typename T>
void copyD2D(T *dst, const T *src, size_t n, cudaStream_t stream = 0) {
  CUDA_CALL(cudaMemcpyAsync(dst, src, n*sizeof(T), cudaMemcpyDeviceToDevice, stream));
}

template <typename T>
void copyD2H(T *dst, const T *src, size_t n, cudaStream_t stream = 0) {
  CUDA_CALL(cudaMemcpyAsync(dst, src, n*sizeof(T), cudaMemcpyDeviceToHost, stream));
}

template <typename T>
void copyH2D(T *dst, const T *src, size_t n, cudaStream_t stream = 0) {
  CUDA_CALL(cudaMemcpyAsync(dst, src, n*sizeof(T), cudaMemcpyHostToDevice, stream));
}

template <typename T>
void copyH2H(T *dst, const T *src, size_t n, cudaStream_t stream = 0) {
  CUDA_CALL(cudaMemcpyAsync(dst, src, n*sizeof(T), cudaMemcpyHostToHost, stream));
}


/**
 * @brief Represents a strongly typed device-side buffer with size and capacity.
 *
 * This class behaves somewhat like vector in terms of storage growth.
 * It doesn't support copy construction/assignment nor indexing (not possible on host-side).
 * It does support common query functions, exponential resize and shrinking.
 * It also provides copy utilities from both host- and device-side sources, with ability
 * to specify CUDA stream.
 */
template <typename T>
struct DeviceBuffer {
  DeviceBuffer() = default;
  DeviceBuffer(DeviceBuffer &&other) {
    *this = other;
  }

  DeviceBuffer &operator=(DeviceBuffer &&other) {
    data_ = std::move(other.data_);
    size_ = other.size_;
    capacity_ = other.capacity_;
    other.size_ = 0;
    other.capacity_ = 0;
    return *this;
  }

  size_t size() const { return size_; }
  size_t size_bytes() const { return sizeof(T) * size_; }
  size_t capacity() const { return capacity_; }

  operator T *() { return data_.get(); }
  operator const T *() const { return data_.get(); }

  T *data() { return data_.get(); }
  const T *data() const { return data_.get(); }

  bool empty() const { return size_ == 0; }

  void clear() { size_ = 0; }
  void free() { size_ = 0; capacity_ = 0; data_.reset(); }

  void shrink_to_fit(cudaStream_t stream = 0) {
    reallocate(size_, stream);
  }

  void from_host(const T *source, size_t count, cudaStream_t stream = 0) {
    clear();
    resize(count);
    copyH2D(data_.get(), source, size(), stream);
  }

  void from_device(const T *source, size_t count, cudaStream_t stream = 0) {
    clear();
    resize(count);
    copyD2D(data_.get(), source, size(), stream);
  }

  template <typename ArrayLike>
  if_array_like<ArrayLike> from_host(const ArrayLike &source, cudaStream_t stream = 0) {
    from_host(&source[0], dali::size(source), stream);
  }

  template <typename ArrayLike>
  if_array_like<ArrayLike> from_device(const ArrayLike &source, cudaStream_t stream = 0) {
    from_device(&source[0], dali::size(source), stream);
  }

  void copy(const DeviceBuffer &src, cudaStream_t stream = 0) {
    clear();
    resize(src.size());
    copyH2D(data_.get(), src.data(), size(), stream);
  }

  void reserve(size_t new_cap, cudaStream_t stream = 0) {
    if (new_cap > capacity_) {
      reallocate(new_cap, stream);
    }
  }

  void resize(size_t new_size, cudaStream_t stream = 0) {
    if (new_size > capacity_) {
      size_t new_cap = 2 * capacity_;
      if (new_size > new_cap)
        new_cap = new_size;
      reallocate(new_cap, stream);
    }
    size_ = new_size;
  }

 private:
  void reallocate(size_t new_cap, cudaStream_t stream) {
    if (new_cap == 0) {
      free();
      return;
    }
    if (size_ == 0) {
      data_.reset();
      capacity_ = size_ = 0;
      data_ = allocate(new_cap);
      capacity_ = new_cap;
    } else {
      auto new_data = allocate(new_cap);
      copyD2D(new_data.get(), data_.get(), size(), stream);
      capacity_ = new_cap;
      data_ = std::move(new_data);
    }
  }

  static std::unique_ptr<T, std::function<void(T*)>> allocate(size_t count) {
    T *ptr = nullptr;
    CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    if (!ptr) {
      (void)cudaGetLastError();
      throw dali::CUDABadAlloc(count * sizeof(T));
    }
    return { ptr, [](T* ptr) { cudaFree(ptr); } };
  }

  std::unique_ptr<T, std::function<void(T*)>> data_;
  size_t capacity_ = 0;
  size_t size_ = 0;
};

}  // namespace dali


#endif  // DALI_CORE_DEV_BUFFER_H_
