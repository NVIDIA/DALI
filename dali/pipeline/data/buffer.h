// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/core/device_guard.h"
#include "dali/core/error_handling.h"
#include "dali/core/util.h"
#include "dali/pipeline/data/types.h"
#include "dali/core/format.h"
#include "dali/core/access_order.h"

namespace dali {

class GPUBackend;
class CPUBackend;


DLL_PUBLIC shared_ptr<uint8_t> AllocBuffer(size_t bytes,
                                           bool pinned, int device_id,
                                           AccessOrder order, GPUBackend *);

DLL_PUBLIC shared_ptr<uint8_t> AllocBuffer(size_t bytes,
                                           bool pinned, int device_id,
                                           AccessOrder order, CPUBackend *);


/**
 * @brief Indicates, based on environment cues, whether pinned memory allocations should be avoided.
 */
DLL_PUBLIC bool RestrictPinnedMemUsage();

template <typename Backend>
inline shared_ptr<uint8_t> AllocBuffer(size_t bytes,
                                       bool pinned, int device_id = -1,
                                       AccessOrder order = {}) {
  return AllocBuffer(bytes, pinned, device_id, order, static_cast<Backend*>(nullptr));
}

DLL_PUBLIC AccessOrder get_deletion_order(const std::shared_ptr<void> &ptr);
DLL_PUBLIC bool set_deletion_order(const std::shared_ptr<void> &ptr, AccessOrder order);

template <typename T>
inline AccessOrder get_deletion_order(const std::shared_ptr<T> &ptr) {
  return get_deletion_order(std::static_pointer_cast<void>(ptr));
}

template <typename T>
inline bool set_deletion_order(const std::shared_ptr<T> &ptr, AccessOrder order) {
  return set_deletion_order(std::static_pointer_cast<void>(ptr), order);
}

/**
 * @brief Base class to provide common functionality needed by Pipeline data
 * structures. Not meant for use, does not provide methods for allocating
 * any actual storage. The 'Backend' template parameter dictates where the
 * underlying storage is located (CPU or GPU).
 *
 * Buffers are untyped on construction, and don't receive a valid type until
 * 'set_type' is called on a non-const buffer. Upon receiving
 * a valid type, the underlying storage for the buffer is allocated. The type
 * of the underlying data can change over the lifetime of an object if
 * 'set_type' is called again where the calling type does not
 * match the underlying type on the buffer. In this case, the Buffer swaps its
 * current type, but only re-allocates memory if it does not have enough bytes
 * of allocated storage to store the number of elements in the buffer with the
 * new data type size.
 *
 * Buffers should be used to store POD types only. No construction or
 * destruction is provided, only raw memory allocation.
 */
template <typename Backend>
class DLL_PUBLIC Buffer {
 public:
  /**
   * @brief Initializes a buffer of size 0.
   */
  inline Buffer() = default;
  inline Buffer(const Buffer &) = delete;
  inline Buffer(Buffer &&other) {
    *this = std::move(other);
  }

  virtual ~Buffer() {
    try {
      free_storage();
    } catch (CUDAError &e) {
      if (!e.is_unloading())
        std::terminate();
    }
  }

  Buffer &operator=(Buffer &&other) {
    move_buffer(std::move(other));
    return *this;
  }

  /**
   * @brief Returns a typed pointer to the underlying storage.
   * The calling type must match the underlying type of the buffer.
   */
  template <typename T>
  inline T* mutable_data() {
    DALI_ENFORCE(type_.id() == TypeTable::GetTypeId<T>(),
                 make_string("Calling type does not match buffer data type, requested type: ",
                 TypeTable::GetTypeName<T>(), " current buffer type: ", type_.name(),
                 ". To set type for the Buffer use 'set_type<T>()' or Resize(shape, type) first."));
    return static_cast<T*>(data_.get());
  }

  /**
   * @brief Returns a const, typed pointer to the underlying storage.
   * The calling type must match the underlying type of the buffer.
   */
  template <typename T>
  inline const T* data() const {
    DALI_ENFORCE(type_.id() == TypeTable::GetTypeId<T>(),
                 make_string("Calling type does not match buffer data type, requested type: ",
                 TypeTable::GetTypeName<T>(), " current buffer type: ", type_.name(),
                 ". To set type for the Buffer use 'set_type<T>()' or Resize(shape, type) first."));
    return static_cast<T*>(data_.get());
  }

  /**
   * @brief Return an un-typed pointer to the underlying storage.
   * A valid type must be set prior to calling this method by calling
   * the non-const version of the method, or calling 'set_type'.
   */
  inline void* raw_mutable_data() {
    return data_.get();
  }

  /**
   * @brief Return a const, un-typed pointer to the underlying storage.
   * A valid type must be set prior to calling this method by calling
   * the non-const version of the method, or calling 'set_type'.
   */
  inline const void* raw_data() const {
    return data_.get();
  }

  /**
   * @brief Returns the size in elements of the underlying data
   */
  inline Index size() const {
    return size_;
  }

  /**
   * @brief Returns the size in bytes of the underlying data
   */
  inline size_t nbytes() const {
    // Note: This returns the number of bytes occupied by the current
    // number of elements stored in the buffer. This is not necessarily
    // the number of bytes of the underlying allocation (num_bytes_)
    return size_ * type_.size();
  }

  /**
   * @brief Returns the real size of the allocation
   */
  inline size_t capacity() const {
    return num_bytes_;
  }

  /**
   * @brief Returns the id of the datatype of the underlying storage.
   */
  inline DALIDataType type() const {
    return type_.id();
  }

  /**
   * @brief Returns the TypeInfo object that keeps track of the
   * datatype of the underlying storage.
   */
  inline const TypeInfo &type_info() const {
    return type_;
  }

  using AllocFunc = std::function<shared_ptr<uint8_t>(size_t)>;

  /**
   * @brief Sets a custom allocation function.
   *
   * Sets a custom allocation function. The allocation function returns
   * a shared pointer with a matching deleter.
   *
   * @remarks Experimental - subject to change
   */
  inline void set_alloc_func(AllocFunc allocate) {
    allocate_ = std::move(allocate);
  }

  /**
   * @brief Returns the current custom allocation function.
   *
   * @return Allocation function. If not set, an empty function object is returned.
   *
   * @remarks Experimental - subject to change
   */
  const AllocFunc &alloc_func() const noexcept {
    return allocate_;
  }

  /**
   * @brief Return true if there was data allocation
   */
  inline bool has_data() const noexcept {
    return !!data_;
  }

  std::shared_ptr<void> get_data_ptr() const {
    return data_;
  }

  /**
   * @brief Sets the type of allocation (pinned/non-pinned) for CPU buffers
   */
  inline void set_pinned(bool pinned) {
    DALI_ENFORCE(!has_data(), "Can only set allocation mode before first allocation");
    DALI_ENFORCE(!allocate_, "Cannot set allocation mode when a custom allocator is used.");
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
   * @brief Sets a device this buffer was allocated on
   * If the backend is CPUBackend, should be -1
   */
  void set_device_id(int device) {
    device_ = device;
  }

  /**
   * @brief Returns the order in which the data is accessed - it can be either host order
   *        or a stream order (or unspecified).
   */
  AccessOrder order() const {
    return order_;
  }

  /**
   * @brief Returns the order in which the underlying storage will be freed. This may
   *        differ from the order returned by `order()` if `set_order()` was called
   *        after the storage has been allocated.
   */
  AccessOrder deletion_order() const {
    return get_deletion_order(data_);
  }

  /**
   * @brief Sets the associated access order.
   *
   * @note The caller must ensure that if `order` represents a CUDA stream, that stream
   *       is alive when this buffer is destroyed. This extends to buffers with which this
   *       one shares data. Use CUDAStreamPool::instance to get streams with indefinite lifetime.
   *
   * @param order       The new access order (stream or host). If the new order doesn't have
   *                    a value, the function has no effect.
   * @param synchronize If true, an appropriate synchronization is inserted between the old
   *                    and the new order. The caller may specify `false` if appropriate
   *                    synchronization is guaranteed by other means.
   */
  void set_order(AccessOrder order, bool synchronize = true) {
    if (!order.has_value())
      return;
    if (!synchronize || !order_.has_value()) {
      order_ = order;
      return;
    }
    if (order == order_)
      return;
    if (has_data())  // if there's no data, we don't need to synchronize
      order.wait(order_);
    order_ = order;
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
  inline void set_type(const DALIDataType new_type_id) {
    DALI_ENFORCE(new_type_id != DALI_NO_TYPE, "new_type must be valid type.");
    if (new_type_id == type_.id()) return;
    const TypeInfo &new_type = TypeTable::GetTypeInfo(new_type_id);

    size_t new_num_bytes = size_ * new_type.size();
    if (shares_data_) {
      DALI_ENFORCE(new_num_bytes == num_bytes_ || new_num_bytes == 0,
                   "Buffer that shares data cannot have size "
                   "different than total underlying allocation");
    }

    type_ = new_type;
    if (new_num_bytes > num_bytes_) {
      reserve(new_num_bytes);
    }
  }

  template <typename T>
  inline void set_type() {
    set_type(TypeTable::GetTypeId<T>());
  }

  /**
   * @brief Reserves at least new_num_bytes of storage.
   *
   * @param new_num_bytes The minimum size, in bytes of the buffer. Note that alignment or
   *                      other conditions may cause actual allocation to be larger.
   *                      If the buffer is already as large as `new_num_bytes`, no allocation
   *                      will occur.
   * @param order         The order (stream or host) in which the deallocation is to occur.
   *                      If possible, the new storage will be allocated asyncrhonously in the
   *                      order specified by this argument.
   *                      After this operation completes, the buffer is associated with this order.
   *                      If order is empty, current order is used.
   *
   * @note For speed, use `buf.reserve(size, order)`
   *       For better memory reusing, use `buf.set_order(order); buf.reserve(size);`
   */
  inline void reserve(size_t new_num_bytes, AccessOrder order = {}) {
    DALI_ENFORCE(!shares_data_,
                 "Cannot reallocate Buffer if it is sharing data. "
                 "Clear the status by `Reset()` first.");

    if (data_ && device_ >= 0 && order.is_device() && order.device_id() != device_)
      DALI_FAIL("Cannot reallocate a buffer on a different device!");

    if (new_num_bytes <= num_bytes_) {
      set_order(order);
      return;
    }

    free_storage();
    if (order) {
      set_order(order);
    } else if (!order_ && pinned_) {
      set_order(AccessOrder::host());
    }

    if (device_ < 0) {
      if (order.is_device() && order.device_id() >= 0) {
        device_ = order.device_id();
      } else {
        // re-allocating: get the device
        if (std::is_same<Backend, GPUBackend>::value || pinned_) {
          device_ = order_.device_id();
          if (device_ < 0)
            CUDA_CALL(cudaGetDevice(&device_));
        } else {
          device_ = CPU_ONLY_DEVICE_ID;
        }
      }
    }

    data_ = allocate_ ? allocate_(new_num_bytes)
                      : AllocBuffer<Backend>(new_num_bytes, pinned_, device_, order_);

    num_bytes_ = new_num_bytes;
  }

  /**
   * @brief Deallocates the data and clears the type.
   *
   * The data, if any, is deallocated.
   *
   * @param order  The order (stream or host) in which the deallocation is to occur.
   *               The order is changed to the provided one after the operation completes.
   *               If order is empty, current order is used.
   *
   * @note Use reset(order) only when it is important that the memory is immediately available
   *       in the context specified by order. Otherwise, call reset() first and then
   *       set_order(order) separately for less synchronization.
   */
  void reset(AccessOrder order = {}) {
    set_order(order);
    free_storage();
    type_ = {};
    allocate_ = {};
    size_ = 0;
    shares_data_ = false;
  }

  void swap(Buffer &buffer) {
    std::swap(type_, buffer.type_);
    std::swap(data_, buffer.data_);
    std::swap(allocate_, buffer.allocate_);
    std::swap(size_, buffer.size_);
    std::swap(num_bytes_, buffer.num_bytes_);
    std::swap(device_, buffer.device_);
    std::swap(shares_data_, buffer.shares_data_);
    std::swap(pinned_, buffer.pinned_);
    std::swap(order_, buffer.order_);
  }

  /**
   * @brief Returns a bool indicating if the list shares its underlying storage.
   */
  inline bool shares_data() const {
    return shares_data_;
  }

  inline void ShareData(const Buffer<Backend> &other) {
    free_storage();
    order_ = other.order_;
    data_ = other.data_;
    size_ = other.size_;
    type_ = other.type_;
    num_bytes_ = other.num_bytes_;
    shares_data_ = num_bytes_ > 0 ? true : false;
    device_ = other.device_id();
  }

  /**
   * @brief Set external memory as the backing memory for this Buffer.
   *
   * Current Buffer will be marked as sharing data, and reallocation of memory will be
   * prohibited until reset() is called.
   *
   * For GPU memory, it is assumed to be associated with current device.
   */
  inline void set_backing_allocation(const shared_ptr<void> &ptr, size_t bytes, bool pinned,
                                     DALIDataType type = DALI_NO_TYPE, size_t size = 0) {
    free_storage();
    type_ = TypeTable::GetTypeInfo(type);
    data_ = ptr;
    allocate_ = {};
    size_ = size;
    shares_data_ = data_ != nullptr;
    num_bytes_ = bytes;
    pinned_ = pinned;
    // setting the allocation, get the device
    if ((std::is_same<Backend, GPUBackend>::value || pinned_) && device_ == CPU_ONLY_DEVICE_ID) {
      CUDA_CALL(cudaGetDevice(&device_));
    } else {
      device_ = CPU_ONLY_DEVICE_ID;
    }
  }

  static void SetGrowthFactor(double factor) {
    assert(factor >= 1.0);
    growth_factor_ = factor;
  }
  static void SetShrinkThreshold(double ratio) {
    assert(ratio >= 0 && ratio <= 1);
    shrink_threshold_ = ratio;
  }
  static double GetGrowthFactor() {
    return growth_factor_;
  }
  static double GetShrinkThreshold() {
    return shrink_threshold_;
  }

  DLL_PUBLIC static constexpr double kMaxGrowthFactor = 4;

  /**
   * @brief Resize the Buffer to hold `new_size` elements of current type.
   */
  inline void resize(Index new_size) {
    resize(new_size, type_);
  }

  /**
   * @brief Resize the Buffer to hold `new_size` elements of type `new_type_id`.
   */
  inline void resize(Index new_size, DALIDataType new_type_id) {
    // don't look up the type unless it's different than current one
    const auto &new_type = new_type_id == type_.id() ? type_ : TypeTable::GetTypeInfo(new_type_id);
    resize(new_size, new_type);
  }

  /**
   * @brief Resize the Buffer to hold `new_size` elements of type `new_type_id`.
   */
  inline void resize(Index new_size, const TypeInfo &new_type) {
    DALI_ENFORCE(new_size >= 0, "Input size less than zero not supported.");
    DALI_ENFORCE(IsValidType(new_type), "Buffer can only be resized with a valid type.");
    // If we use NoType the result will always be 0
    size_t new_num_bytes = new_size * new_type.size();

    if (shares_data_) {
      DALI_ENFORCE(new_num_bytes <= num_bytes_,
                   "Cannot change size of a Buffer if it is sharing data. "
                   "Clear the status by `Reset()` first.");
    }

    size_ = new_size;
    type_ = new_type;

    if (shares_data_)
      return;

    if (new_size == 0) {
      if (std::is_same<Backend, GPUBackend>::value && device_ == CPU_ONLY_DEVICE_ID) {
        CUDA_CALL(cudaGetDevice(&device_));
      }
      return;
    }

    if (new_num_bytes > num_bytes_) {
      size_t grow = num_bytes_ * growth_factor_;
      if (grow > new_num_bytes) new_num_bytes = grow;
      reserve(new_num_bytes);
    } else if (!is_pinned() && new_num_bytes < num_bytes_ * shrink_threshold_) {
      free_storage();
      reserve(new_num_bytes);
    }
  }

 protected:
  void move_buffer(Buffer &&buffer) {
    swap(buffer);
    buffer.reset();
  }

  void free_storage(AccessOrder order = {}) {
    if (data_) {
      if (!order)
        order = order_;
      if (!set_deletion_order(data_, order))
        get_deletion_order(data_).wait(order);
      data_.reset();
    }
    num_bytes_ = 0;
  }

  template <typename>
  friend class TensorList;

  static double growth_factor_;
  static double shrink_threshold_;

  static bool default_pinned();

  TypeInfo type_ = {};               // Data type of underlying storage
  shared_ptr<void> data_ = nullptr;  // Pointer to underlying storage
  AllocFunc allocate_;               // Custom allocation function
  Index size_ = 0;                   // The number of elements in the buffer
  size_t num_bytes_ = 0;             // To keep track of the true size of the underlying allocation
  int device_ = CPU_ONLY_DEVICE_ID;  // device the buffer was allocated on
  AccessOrder order_;                // The order of memory access (host or device)
  bool shares_data_ = false;         // Whether we aren't using our own allocation
  bool pinned_ = !RestrictPinnedMemUsage();  // Whether the allocation uses pinned memory
};

template <typename Backend>
DLL_PUBLIC double Buffer<Backend>::growth_factor_ = 1.1;

template <typename Backend>
DLL_PUBLIC double Buffer<Backend>::shrink_threshold_ =
  std::is_same<Backend, CPUBackend>::value ? 0.5 : 0;

template <typename Backend>
constexpr double Buffer<Backend>::kMaxGrowthFactor;


// Macro so we don't have to list these in all
// classes that derive from Buffer
#define USE_BUFFER_MEMBERS()           \
  using Buffer<Backend>::resize;       \
  using Buffer<Backend>::reset;        \
  using Buffer<Backend>::type_;        \
  using Buffer<Backend>::data_;        \
  using Buffer<Backend>::size_;        \
  using Buffer<Backend>::shares_data_; \
  using Buffer<Backend>::num_bytes_;   \
  using Buffer<Backend>::device_;      \
  using Buffer<Backend>::pinned_;      \
  using Buffer<Backend>::move_buffer;  \
  using Buffer<Backend>::free_storage; \
  using Buffer<Backend>::order_;       \


}  // namespace dali

#endif  // DALI_PIPELINE_DATA_BUFFER_H_
