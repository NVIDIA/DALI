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

#ifndef DALI_CORE_UNIQUE_HANDLE_H_
#define DALI_CORE_UNIQUE_HANDLE_H_

#include <utility>
#include "dali/core/common.h"

namespace dali {

/**
 * @brief This class is an analogue of `unique_ptr` for non-memory resource handles.
 *
 * UniqueHandle is a base class for implementing managed resources (files, OS handles, etc).
 * This class provides construction, assigment and decay to underlying handle type as well
 * as equality comparison operators.
 *
 * @tparam HandleType  type of the handle, e.g. `int` for file descriptors or `FILE*` for buffers
 * @tparam Actual      derived class (if using CRTP) or a handle information class.
 *
 * The interface of the `Actual` type:
 * ```
 * static void DestroyHandle(HandleType h);    // free or un-reference the underlying resource
 *
 * static constexpr HandleType null_handle();  // return a null handle; when using CRTP it's
 *                                             // optional and default implementation returns
 *                                             // default-constructed handle value.
 * ```
 *
 * The handle can be populated by either the explicit constructor or using
 * @link reset(handle_type) reset @endlink
 * function. The derived classes may provide other ways of constructing the handle or the entire
 * handle wrapper object.
 *
 * @see CUDAEvent
 * @see CUDAStream
 */
template <typename HandleType, typename Actual>
class UniqueHandle {
 public:
  using handle_type = HandleType;

  constexpr inline UniqueHandle() : handle_(Actual::null_handle()) {}

  /// @brief Constructs a handle wrapper, assuming ownership of given handle.
  constexpr explicit UniqueHandle(handle_type handle) : handle_(handle) {}

  UniqueHandle(const UniqueHandle &) = delete;

  UniqueHandle &operator=(const UniqueHandle &) = delete;

  inline UniqueHandle(UniqueHandle &&other) : handle_(other.handle_) {
    other.handle_ = Actual::null_handle();
  }

  inline UniqueHandle &operator=(UniqueHandle &&other) {
    std::swap(handle_, other.handle_);
    other.reset();
    return *this;
  }

  /// @brief Make the wrapper usable in most context in which the handle type can be used.
  constexpr operator handle_type() const noexcept { return handle_; }

  /**
   * @brief Destroys the underlying resource and resets the handle to null value.
   *
   * @remarks
   * * If the handle is already null, this function is a no-op.
   * * The null value to replace the handle with, is taken from `Actual::null_value()`.
   */
  inline void reset() {
    if (!Actual::is_null_handle(handle_)) {
      Actual::DestroyHandle(handle_);
      handle_ = Actual::null_handle();
    }
  }

  /**
   * @brief Replaces the managed handle by the new one and destroying the old handle.
   * @remarks If `handle` is equal to the currently managed handle, this function is no-op
   */
  inline void reset(handle_type handle) {
    if (handle != handle_) {
      reset();
      handle_ = handle;
    }
  }

  /**
   * @brief Relinquishes the ownership of the handle.
   *
   * The function replaces the managed handle with a null value and returns the old value.
   *
   * @returns The old handle value, no longer managed by UniqueHandle
   * @remarks The null value to replace the handle with, is taken from `Actual::null_value()`.
   */
  inline handle_type release() noexcept {
    handle_type old = handle_;
    handle_ = Actual::null_handle();
    return old;
  }

  /// @brief Indicates whether the handle is non-null.
  constexpr explicit operator bool() const noexcept {
    return !Actual::is_null_handle(handle_);
  }

  static constexpr handle_type null_handle() noexcept { return {}; }
  static constexpr bool is_null_handle(const handle_type& handle) noexcept {
    return handle == Actual::null_handle();
  }

 protected:
  inline ~UniqueHandle() { reset(); }

  handle_type handle_;
};

/**
 * A macro to inherit the common interface from UniqueHandle
 * - useful when using UniqueHandle in CRTP
 */
#define DALI_INHERIT_UNIQUE_HANDLE(HandleType, WrapperClass)\
  using dali::UniqueHandle<HandleType, WrapperClass>::UniqueHandle;\
  using dali::UniqueHandle<HandleType, WrapperClass>::operator=;\

}  // namespace dali

#endif  // DALI_CORE_UNIQUE_HANDLE_H_
