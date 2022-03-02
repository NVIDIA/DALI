// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/mm/memory.h"

namespace dali {

template <typename T>
void copyD2D(T *dst, const T *src, size_t n, AccessOrder order = {}) {
  if (order.is_device())
    CUDA_CALL(cudaMemcpyAsync(dst, src, n*sizeof(T), cudaMemcpyDeviceToDevice, order.stream()));
  else
    CUDA_CALL(cudaMemcpy(dst, src, n*sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
void copyD2H(T *dst, const T *src, size_t n, AccessOrder order = {}) {
  if (order.is_device())
    CUDA_CALL(cudaMemcpyAsync(dst, src, n*sizeof(T), cudaMemcpyDeviceToHost, order.stream()));
  else
    CUDA_CALL(cudaMemcpy(dst, src, n*sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void copyH2D(T *dst, const T *src, size_t n, AccessOrder order = {}) {
  if (order.is_device())
    CUDA_CALL(cudaMemcpyAsync(dst, src, n*sizeof(T), cudaMemcpyHostToDevice, order.stream()));
  else
    CUDA_CALL(cudaMemcpy(dst, src, n*sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void copyH2H(T *dst, const T *src, size_t n, AccessOrder order = {}) {
  if (order.is_device())
    CUDA_CALL(cudaMemcpyAsync(dst, src, n*sizeof(T), cudaMemcpyHostToHost, order.stream()));
  else
    memcpy(dst, src, n*sizeof(T));
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

  void shrink_to_fit(AccessOrder order = {}) {
    reallocate(size_, order);
  }

  void from_host(const T *source, size_t count, AccessOrder order = {}) {
    clear();
    resize(count);
    copyH2D(data_.get(), source, size(), order);
  }

  void from_device(const T *source, size_t count, AccessOrder order = {}) {
    clear();
    resize(count);
    copyD2D(data_.get(), source, size(), order);
  }

  template <typename ArrayLike>
  if_array_like<ArrayLike> from_host(const ArrayLike &source, AccessOrder order = {}) {
    from_host(&source[0], dali::size(source), order);
  }

  template <typename ArrayLike>
  if_array_like<ArrayLike> from_device(const ArrayLike &source, AccessOrder order = {}) {
    from_device(&source[0], dali::size(source), order);
  }

  void copy(const DeviceBuffer &src, AccessOrder order = {}) {
    clear();
    resize(src.size());
    copyD2D(data_.get(), src.data(), size(), order);
  }

  void reserve(size_t new_cap, AccessOrder order = {}) {
    if (new_cap > capacity_) {
      reallocate(new_cap, order);
    }
  }

  void resize(size_t new_size, AccessOrder order = {}) {
    if (new_size > capacity_) {
      size_t new_cap = 2 * capacity_;
      if (new_size > new_cap)
        new_cap = new_size;
      reallocate(new_cap, order);
    }
    size_ = new_size;
  }

 private:
  void reallocate(size_t new_cap, AccessOrder order) {
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
      copyD2D(new_data.get(), data_.get(), size(), order);
      capacity_ = new_cap;
      data_ = std::move(new_data);
    }
  }

  static mm::uptr<T> allocate(size_t count) {
    return mm::alloc_raw_unique<T, mm::memory_kind::device>(count);
  }

  mm::uptr<T> data_;
  size_t capacity_ = 0;
  size_t size_ = 0;
};

}  // namespace dali


#endif  // DALI_CORE_DEV_BUFFER_H_
