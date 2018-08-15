// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_DATA_BUFFER_H_
#define DALI_PIPELINE_DATA_BUFFER_H_

#include <limits>
#include <numeric>
#include <functional>
#include <vector>
#include <string>
#include <memory>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/data/types.h"

namespace dali {

class GPUBackend;

// Helper function to get product of dims
inline Index Product(const vector<Index> &shape) {
  if (shape.size() == 0) return 0;
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<Index>());
}

// Helper function to get a string of the data shape
inline string ShapeString(vector<Index> shape) {
  string tmp;
  for (auto &val : shape) tmp += std::to_string(val) + " ";
  return tmp;
}

// NOTE: Data storage types in DALI use delayed allocation, and have a
// small custom type system that allows us to circumvent template
// paramters. This is turn allows the Pipeline to manage all intermediate
// memory, opening the door for optimizations and reducing the work that
// must be done by the user when defining operations.

/**
 * @brief Base class to provide common functionality needed by Pipeline data
 * structures. Not meant for use, does not provide methods for allocating
 * any actual storage. The 'Backend' template parameter dictates where the
 * underlying storage is located (CPU or GPU).
 *
 * Buffers are untyped on construction, and don't receive a valid type until
 * 'set_type' or 'data<T>()' is called on a non-const buffer. Upon receiving
 * a valid type, the underlying storage for the buffer is allocated. The type
 * of the underlying data can change over the lifetime of an object if
 * 'set_type' or 'data<T>()' is called again where the calling type does not
 * match the underlying type on the buffer. In this case, the Buffer swaps its
 * current type, but only re-allocates memory if it does not have enough bytes
 * of allocated storage to store the number of elements in the buffer with the
 * new data type size.
 */
template <typename Backend>
class Buffer {
 public:
  /**
   * @brief Initializes a buffer of size 0.
   */
  inline Buffer() : data_(nullptr),
                    size_(0),
                    shares_data_(false),
                    num_bytes_(0),
                    pinned_(true),
                    device_(-1)
    {}

  virtual ~Buffer()// = default;
  {
    if (std::is_same<Backend, GPUBackend>::value) {
    std::cout << "Freeing buffer holding " << num_bytes_ << " bytes" << std::endl;
    }
  }

  /**
   * @brief Returns a typed pointer to the underlying storage.
   */
  template <typename T>
  inline T* mutable_data() {
    return static_cast<T*>(data_.get());
  }

  /**
   * @brief Returns a const, typed pointer to the underlying storage.
   */
  template <typename T>
  inline const T* data() const {
    return static_cast<T*>(data_.get());
  }

  /**
   * @brief Return an un-typed pointer to the underlying storage.
   */
  inline void* raw_mutable_data() {
    return static_cast<void*>(data_.get());
  }

  /**
   * @brief Return an const, un-typed pointer to the underlying storage.
   */
  inline const void* raw_data() const {
    return static_cast<void*>(data_.get());
  }

  /**
   * @brief Returns the current size of the buffer in bytes.
   */
  inline Index size() const { return size_; }

  /**
   * @brief Returns the real size of the allocation.
   */
  inline size_t capacity() const {
    return num_bytes_;
  }

  /**
   * @brief Sets the type of allocation (pinned/non-pinned) for
   * CPU buffers
   */
  inline void set_pinned(const bool pinned) {
    DALI_ENFORCE(!data_, "Can only set allocation mode before first allocation");
    pinned_ = pinned;
  }

  inline bool is_pinned() const {
    return pinned_;
  }

  /**
   * @brief Returns a device this buffer was allocated on
   * If the backend is CPUBackend, return -1
   */
  int device_id() const {
    return device_;
  }

  /**
   * @brief Returns a bool indicating if the list shares its underlying storage.
   */
  inline bool shares_data() const { return shares_data_; }

  // Helper function for cleaning up data storage. This unfortunately
  // has to be public so that we can bind it into the deleter of our
  // shared pointers
  void DeleterHelper(void *ptr, Index size) {
    // change to correct device for deletion
    // Note: Can't use device guard due to potentially not GPUBackend.
    int current_device = 0;
    if (std::is_same<Backend, GPUBackend>::value) {
      auto err = cudaGetDevice(&current_device);

      // It's possible that the CUDA driver is unloading / has unloaded
      // before we get to deleting all buffers. In that case we catch the appropriate
      // error code and simply return.
      if (err == cudaErrorCudartUnloading) return;

      CUDA_CALL(cudaSetDevice(device_));
    }
    if (ptr) {
      // Only deallocate, underlying data freed elsewhere
      Backend::Delete(ptr, size, pinned_);
    }

    // reset to original calling device for consistency
    if (std::is_same<Backend, GPUBackend>::value) {
      CUDA_CALL(cudaSetDevice(current_device));
    }
  }

  /**
   * @brief Resize the buffer.
   * Returns whether the underlying allocation
   * changed.
   */
  bool Resize(size_t new_size) {
    return ResizeHelper(new_size);
  }

  void ShareData(void *ptr, size_t bytes, int device_id) {
    data_.reset(ptr, [](void *) {});
    num_bytes_ = bytes;
    size_ = 0;
    device_ = device_id;

    shares_data_ = num_bytes_ > 0 ? true : false;
  }

  DISABLE_COPY_MOVE_ASSIGN(Buffer);

  shared_ptr<void> data_;  // Pointer to underlying storage
 protected:
  // Helper to resize the underlying allocation
  inline bool ResizeHelper(Index new_size) {
    DALI_ENFORCE(new_size >= 0,
        "Size of buffer has to be positive, got " + to_string(new_size));

    size_ = new_size;
    if ((size_t)new_size > num_bytes_) {
      new_size *= alloc_mult;
      // re-allocating: get the device
      if (std::is_same<Backend, GPUBackend>::value) {
        CUDA_CALL(cudaGetDevice(&device_));
      }
      data_.reset(Backend::New(new_size, pinned_), std::bind(
              &Buffer<Backend>::DeleterHelper,
              this, std::placeholders::_1,
              new_size));
      num_bytes_ = new_size;

      // If we were sharing data, we aren't anymore
      shares_data_ = false;

      return true;
    }
    return false;
  }

  const double alloc_mult = 1.0;

  Backend backend_;

  Index size_;  // current size of the buffer
  bool shares_data_;

  size_t num_bytes_;  // the size of the allocation

  bool pinned_;  // Whether the allocation uses pinned memory

  // device the buffer was allocated on
  int device_;
};

// Macro so we don't have to list these in all
// classes that derive from Buffer
#define USE_BUFFER_MEMBERS()                    \
  using Buffer<Backend>::ResizeHelper;          \
  using Buffer<Backend>::backend_;              \
  using Buffer<Backend>::data_;                 \
  using Buffer<Backend>::size_;                 \
  using Buffer<Backend>::shares_data_;          \
  using Buffer<Backend>::num_bytes_;            \
  using Buffer<Backend>::device_

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_BUFFER_H_
