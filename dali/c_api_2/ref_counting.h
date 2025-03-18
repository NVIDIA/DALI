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

#ifndef DALI_C_API_2_REF_COUNTING_H_
#define DALI_C_API_2_REF_COUNTING_H_

#include <atomic>
#include <type_traits>
#include <utility>

namespace dali::c_api {

/** A base class for objects that feature intrusive reference counting. */
class RefCountedObject {
 public:
  /** Increments the reference count and returns the new value. */
  int IncRef() noexcept {
    return std::atomic_fetch_add_explicit(&ref_, 1, std::memory_order_relaxed) + 1;
  }

  /** Decrements the reference count and returns the new value.
   *
   * When the reference count reaches 0, the object is deleted.
   */
  int DecRef() noexcept {
    int ret = std::atomic_fetch_sub_explicit(&ref_, 1, std::memory_order_acq_rel) - 1;
    if (!ret)
        delete this;
    return ret;
  }

  /** Returns the current reference count. */
  int RefCount() const noexcept {
    return ref_.load(std::memory_order_relaxed);
  }

  virtual ~RefCountedObject() = default;

 private:
  std::atomic<int> ref_{1};
};

/** A smart pointer that manages an object with intrusive reference counting. */
template <typename T>
class RefCountedPtr {
 public:
  constexpr RefCountedPtr() noexcept = default;

  /** Constructs a RefCountedPtr from a pointer to an object.
   *
   * @param ptr The pointer to the object.
   * @param inc_ref If true, the reference count is incremented.
   *                If false, the ownership of the object is transferred to the RefCountedPtr.
   */
  explicit RefCountedPtr(T *ptr, bool inc_ref = false) noexcept : ptr_(ptr) {
    if (inc_ref && ptr_)
      ptr_->IncRef();
  }

  ~RefCountedPtr() {
    reset();
  }

  RefCountedPtr(const RefCountedPtr<T> &other) noexcept : ptr_(other.ptr_) {
    if (ptr_)
      ptr_->IncRef();
  }

  template <typename U, std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
  RefCountedPtr(const RefCountedPtr<U> &other) noexcept : ptr_(other.ptr_) {
    if (ptr_)
      ptr_->IncRef();
  }

  template <typename U, std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
  RefCountedPtr(RefCountedPtr<U> &&other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }

  RefCountedPtr &operator=(const RefCountedPtr &other) noexcept {
    return this->operator= <T>(other);
  }


  template <typename U>
  std::enable_if_t<std::is_convertible_v<U *, T *>, RefCountedPtr> &
  operator=(const RefCountedPtr<U> &other) noexcept {
    if (ptr_ == other.ptr_)
      return *this;
    if (other.ptr_)
      other.ptr_->IncRef();
    if (ptr_)
      ptr_->DecRef();
    ptr_ = other.ptr_;
    return *this;
  }

  RefCountedPtr &operator=(RefCountedPtr &&other) noexcept {
    return this->operator= <T>(std::move(other));
  }

  template <typename U>
  std::enable_if_t<std::is_convertible_v<U *, T *>, RefCountedPtr> &
  operator=(RefCountedPtr &&other) noexcept {
    if (&other == this)
      return *this;
    std::swap(ptr_, other.ptr_);
    other.reset();
    return *this;
  }

  void reset() noexcept {
    if (ptr_)
      ptr_->DecRef();
    ptr_ = nullptr;
  }

  [[nodiscard]] T *release() noexcept {
    T *p = ptr_;
    ptr_ = nullptr;
    return p;
  }

  constexpr T *operator->() const & noexcept { return ptr_; }

  constexpr T &operator*() const & noexcept { return *ptr_; }

  constexpr T *get() const & noexcept { return ptr_; }

  explicit constexpr operator bool() const noexcept { return ptr_ != nullptr; }

 private:
  template <typename U>
  friend class RefCountedPtr;
  T *ptr_ = nullptr;
};

}  // namespace dali::c_api

#endif  // DALI_C_API_2_REF_COUNTING_H_

