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

#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/data/types.h"
#include "dali/kernels/util.h"

namespace dali {

class GPUBackend;

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
 *
 * Buffers should be used to store POD types only. No construction or
 * destruction is provided, only raw memory allocation.
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

  virtual ~Buffer() = default;

  /**
   * @brief Returns a typed pointer to the underlying storage. If the
   * buffer has not been allocated because it does not yet have a type,
   * the calling type is taken to be the type of the data and the memory
   * is allocated.
   *
   * If the buffer already has a valid type, and the calling type does
   * not match, the type of the buffer is reset and the underlying
   * storage is re-allocated if the buffer does not currently own
   * enough memory to store the current number of elements with the
   * new data type.
   */
  template <typename T>
  inline T* mutable_data() {
    // Note: Call to 'set_type' will immediately return if the calling
    // type matches the current type of the buffer.
    TypeInfo calling_type;
    calling_type.SetType<T>();
    set_type(calling_type);
    return static_cast<T*>(data_.get());
  }

  /**
   * @brief Returns a const, typed pointer to the underlying storage.
   * The calling type must match the underlying type of the buffer.
   */
  template <typename T>
  inline const T* data() const {
    DALI_ENFORCE(IsValidType(type_),
        "Buffer has no type, 'mutable_data<T>()' must be called "
        "on non-const buffer to set valid type for " + type_.name());
    DALI_ENFORCE(type_.id() == TypeTable::GetTypeID<T>(),
        "Calling type does not match buffer data type: " +
        TypeTable::GetTypeName<T>() + " v. " + type_.name());
    return static_cast<T*>(data_.get());
  }

  /**
   * @brief Return an un-typed pointer to the underlying storage.
   * A valid type must be set prior to calling this method by calling
   * the non-const version of the method, or calling 'set_type'.
   */
  inline void* raw_mutable_data() {
    // Empty tensor
    if (data_ == nullptr)
      return nullptr;
    DALI_ENFORCE(IsValidType(type_),
        "Buffer has no type, 'mutable_data<T>()' or 'set_type' must "
        "be called on non-const buffer to set valid type");
    return static_cast<void*>(data_.get());
  }

  /**
   * @brief Return an const, un-typed pointer to the underlying storage.
   * A valid type must be set prior to calling this method by calling
   * the non-const version of the method, or calling 'set_type'.
   */
  inline const void* raw_data() const {
    // Empty tensor
    if (data_ == nullptr)
      return nullptr;
    DALI_ENFORCE(IsValidType(type_),
        "Buffer has no type, 'mutable_data<T>()' or 'set_type' must "
        "be called on non-const buffer to set valid type");
    return static_cast<void*>(data_.get());
  }

  /**
   * @brief Returns the size in elements of the underlying data
   */
  inline Index size() const { return size_; }

  /**
   * @brief Returns the size in bytes of the underlying data
   */
  inline size_t nbytes() const {
    // Note: This returns the number of bytes occupied by the current
    // number of elements stored in the buffer. This is not neccessarily
    // the number of bytes of the underlying allocation (num_bytes_)
    return size_*type_.size();
  }

  /**
   * @brief Returns the real size of the allocation
   */
  inline size_t capacity() const {
    return num_bytes_;
  }

  /**
   * @brief Returns the padding value of allocations caused by Resize() call
   */
  static inline size_t padding() {
    return kPaddding;
  }

  /**
   * @brief Returns the TypeInfo object that keeps track of the
   * datatype of the underlying storage.
   */
  inline TypeInfo type() const {
    return type_;
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
   * @brief Sets the type of the buffer. If the buffer has not been
   * allocated because it does not yet have a type, the calling type
   * is taken to be the type of the data and the memory is allocated.
   *
   * If the buffer already has a valid type, and the calling type does
   * not match, the type of the buffer is reset and the underlying
   * storage is re-allocated if the buffer does not currently own
   * enough memory to store the current number of elements with the
   * new data type.
   */
  inline void set_type(const TypeInfo &new_type) {
    DALI_ENFORCE(IsValidType(new_type), "new_type must be valid type.");
    if (new_type == type_) return;
    type_ = new_type;

    size_t new_num_bytes = size_ * type_.size();
    if (new_num_bytes > num_bytes_) {
      reserve(new_num_bytes);
    }
  }

  inline void reserve(size_t new_num_bytes) {
    if (new_num_bytes <= num_bytes_)
      return;

    // re-allocating: get the device
    if (std::is_same<Backend, GPUBackend>::value) {
      CUDA_CALL(cudaGetDevice(&device_));
    }

    data_.reset(
      Backend::New(new_num_bytes, pinned_),
      std::bind(FreeMemory, std::placeholders::_1, new_num_bytes, device_, pinned_));

    num_bytes_ = new_num_bytes;

    // If we were sharing data, we aren't anymore
    shares_data_ = false;
  }

  inline void free() {
    data_.reset();
    num_bytes_ = 0;
  }

  /**
   * @brief Returns a bool indicating if the list shares its underlying storage.
   */
  inline bool shares_data() const { return shares_data_; }

  DISABLE_COPY_MOVE_ASSIGN(Buffer);

 protected:
  static void FreeMemory(void *ptr, size_t bytes, int device, bool pinned) {
    // change to correct device for deletion
    // Note: Can't use device guard due to potentially not GPUBackend.
    int current_device = 0;
    if (std::is_same<Backend, GPUBackend>::value) {
      CUDA_CALL(cudaGetDevice(&current_device));
      CUDA_CALL(cudaSetDevice(device));
    }
    Backend::Delete(ptr, bytes, pinned);

    // reset to original calling device for consistency
    if (std::is_same<Backend, GPUBackend>::value) {
      CUDA_CALL(cudaSetDevice(current_device));
    }
  }

  // Helper to resize the underlying allocation
  inline void ResizeHelper(Index new_size) {
    DALI_ENFORCE(new_size >= 0, "Input size less than zero not supported.");

    size_ = new_size;

    if (new_size == 0) {
      if (std::is_same<Backend, GPUBackend>::value && device_ == -1) {
        CUDA_CALL(cudaGetDevice(&device_));
      }
      return;
    }

    if (!IsValidType(type_)) {
      return;
    }

    size_t new_num_bytes = new_size * type_.size();
    if (new_num_bytes > num_bytes_) {
      size_t grow = num_bytes_*kAllocMult;
      grow = (grow + kPaddding) & ~(kPaddding - 1);
      if (grow > new_num_bytes)
        new_num_bytes = grow;
      reserve(new_num_bytes);
    }
  }

  const double kAllocMult = 1.0;
  // round to 1kB
  static constexpr size_t kPaddding = 1024;

  Backend backend_;

  TypeInfo type_;  // Data type of underlying storage
  shared_ptr<void> data_;  // Pointer to underlying storage
  Index size_;  // The number of elements in the buffer
  bool shares_data_;

  // To keep track of the true size
  // of the underlying allocation
  size_t num_bytes_;

  bool pinned_;  // Whether the allocation uses pinned memory

  // device the buffer was allocated on
  int device_;
};

// Macro so we don't have to list these in all
// classes that derive from Buffer
#define USE_BUFFER_MEMBERS()                    \
  using Buffer<Backend>::ResizeHelper;          \
  using Buffer<Backend>::backend_;              \
  using Buffer<Backend>::type_;                 \
  using Buffer<Backend>::data_;                 \
  using Buffer<Backend>::size_;                 \
  using Buffer<Backend>::shares_data_;          \
  using Buffer<Backend>::num_bytes_;            \
  using Buffer<Backend>::device_

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_BUFFER_H_
