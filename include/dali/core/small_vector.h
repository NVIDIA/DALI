// Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_SMALL_VECTOR_H_
#define DALI_CORE_SMALL_VECTOR_H_

#include <cuda_runtime.h>
#include <cstring>
#include <utility>
#include <memory>
#include <vector>
#include "dali/core/util.h"
#include "dali/core/cuda_utils.h"

#pragma GCC diagnostic push
#if __GNUC__ > 11
  // most recent gcc seems to be confused by some things in small vector raising false warnings
  #if defined(__has_warning)
    #if __has_warning("-Wuse-after-free")
      #pragma GCC diagnostic ignored "-Wuse-after-free"
    #endif
  #else
    #pragma GCC diagnostic ignored "-Wuse-after-free"
  #endif
#endif

namespace dali {

template <typename T, typename Allocator, bool Contextless = std::is_empty<Allocator>::value>
struct SmallVectorAlloc {
  SmallVectorAlloc() = default;
  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV SmallVectorAlloc(Allocator &&allocator) : allocator_(cuda_move(allocator)) {}
  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV SmallVectorAlloc(const Allocator &allocator) : allocator_(allocator) {}

  Allocator allocator_;
  __host__ T *allocate(size_t count) {
    return allocator_.allocate(count);
  }
  __host__ void deallocate(T *ptr, size_t count) {
    allocator_.deallocate(ptr, count);
  }
};

template <typename T, typename Allocator>
struct SmallVectorAlloc<T, Allocator, true> {
  static __host__ T *allocate(size_t count) {
    return Allocator().allocate(count);
  }
  static __host__ void deallocate(T *ptr, size_t count) {
    Allocator().deallocate(ptr, count);
  }
  // Add no-op __device__ overload for functions used in __host__ __device__ context by clang
  #if defined(__clang__) && defined(__CUDA__)
  static __device__ T *allocate(size_t count) {
    return nullptr;
  }
  static __device__ void deallocate(T *ptr, size_t count) {}
  #endif
};

template <typename T>
struct SmallVectorAlloc<T, device_side_allocator<T>, true> {
  static __device__ T *allocate(size_t count) {
    return device_side_allocator<T>::allocate(count);
  }
  static __device__ void deallocate(T *ptr, size_t count) {
    device_side_allocator<T>::deallocate(ptr, count);
  }
};


template <typename T, bool is_pod = is_pod_v<T>>
class SmallVectorBase {
 protected:
  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV static void move_and_destroy(T *dest, T *src, size_t count) noexcept {
    for (size_t i = 0; i < count; i++) {
      new(dest + i) T(cuda_move(src[i]));
      src[i].~T();
    }
  }

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV static void destroy(T *ptr, size_t count) noexcept {
    for (size_t i = 0; i < count; i++) {
      ptr[i].~T();
    }
  }

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV static void copy(T *dst, const T *src, size_t count) {
    for (size_t i = 0; i <count; i++)
      dst[i] = src[i];
  }
};


template <typename T>
class SmallVectorBase<T, true> {
 protected:
  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV void copy(T *dst, const T *src, size_t count) noexcept {
#ifdef __CUDA_ARCH__
    for (size_t i = 0; i < count; i++) {
      dst[i] = src[i];
    }
#else
    memcpy(dst, src, count * sizeof(T));
#endif
  }
  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV void move_and_destroy(T *dst, T *src, size_t count) noexcept {
    copy(dst, src, count);
  }

  DALI_HOST_DEV static void destroy(T *, size_t) noexcept {}
};

#if defined(__clang__) && defined(__CUDA__)
template <typename T>
__device__ device_side_allocator<T> GetDefaultAlocatorType() { return {}; }

template <typename T>
__host__ std::allocator<T> GetDefaultAlocatorType() { return {}; }

template <typename T>
using default_small_vector_allocator =
    std::remove_reference_t<decltype(GetDefaultAlocatorType<T>())>;

#else
#ifdef __CUDA_ARCH__
template <typename T>
using default_small_vector_allocator = device_side_allocator<T>;
#else
template <typename T>
using default_small_vector_allocator = std::allocator<T>;
#endif

#endif

template <typename T, size_t static_size_, typename allocator = default_small_vector_allocator<T>>
class SmallVector : SmallVectorAlloc<T, allocator>, SmallVectorBase<T> {
  using Alloc = SmallVectorAlloc<T, allocator>;

 public:
  static constexpr const size_t static_size = static_size_;  // NOLINT (kOnstant)
  DALI_HOST_DEV SmallVector() : dynamic{} {}

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV explicit SmallVector(allocator &&alloc) : Alloc(cuda_move(alloc)), dynamic{} {}

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV explicit SmallVector(const allocator &alloc) : Alloc(alloc), dynamic{} {}

  DALI_HOST_DEV SmallVector(const T *data, size_t count) : SmallVector() {
    copy_assign(data, count);
  }

  DALI_NO_EXEC_CHECK
  template <typename Iterator>
  DALI_HOST_DEV SmallVector(Iterator begin, Iterator end) : SmallVector() {
    copy_assign(begin, end);
  }

  __host__ SmallVector(const std::vector<T> &v) : SmallVector() {
    *this = v;
  }


  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV SmallVector(std::initializer_list<T> il) : SmallVector() {
    auto *data = &*il.begin();
    auto count = il.end() - il.begin();
    copy_assign(data, count);
  }

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV ~SmallVector() {
    T *ptr = data();
    this->destroy(ptr, size());
    if (is_dynamic())
      deallocate(ptr, capacity());
    set_size(0);
    reset_capacity();
  }

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV SmallVector(const SmallVector &other) : SmallVector() {
    *this = other;
  }

  DALI_NO_EXEC_CHECK
  template <size_t other_static_size, typename alloc>
  DALI_HOST_DEV SmallVector(const SmallVector<T, other_static_size, alloc> &other)
  : SmallVector() {
    *this = other;
  }

  template <size_t other_static_size>
  DALI_HOST_DEV SmallVector(SmallVector<T, other_static_size, allocator> &&other) noexcept
  : SmallVector() {
    *this = cuda_move(other);
  }

  DALI_HOST_DEV SmallVector &operator=(const SmallVector &v) {
    copy_assign(v);
    return *this;
  }

  DALI_NO_EXEC_CHECK
  template <size_t other_static_size, typename alloc>
  DALI_HOST_DEV SmallVector &operator=(const SmallVector<T, other_static_size, alloc> &v) {
    copy_assign(v);
    return *this;
  }

  DALI_NO_EXEC_CHECK
  template <typename Collection>
  DALI_HOST_DEV if_array_like<Collection, SmallVector &> operator=(const Collection &c) {
    auto n = dali::size(c);
    clear();
    reserve(n);
    T *ptr = data();
    for (decltype(n) i = 0; i < n; i++) {
      new(ptr+i) T(c[i]);
      if (!noexcept(new(ptr+i) T(c[i])))
        set_size(i+1);
    }
    set_size(n);
    return *this;
  }

  DALI_NO_EXEC_CHECK
  template <typename Iterator>
  DALI_HOST_DEV void copy_assign(Iterator begin, Iterator end) {
    auto n = end - begin;
    clear();
    reserve(n);
    for (Iterator i = begin; i != end; i++) {
      push_back(*i);
    }
  }

  DALI_HOST_DEV void copy_assign(const T *begin, const T *end) {
    copy_assign(begin, end-begin);
  }

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV void copy_assign(const T *data, size_t count) {
    clear();
    reserve(count);
    T *ptr = this->data();
    if (is_pod_v<T>) {
      this->copy(ptr, data, count);
      set_size(count);
    } else {
      for (size_t i = 0; i < count; i++) {
        new(ptr+i) T(data[i]);
        if (!noexcept(new(ptr+i) T(data[i])))
          set_size(i+1);
      }
      set_size(count);
    }
  }

  DALI_NO_EXEC_CHECK
  template <size_t other_static_size, typename alloc>
  DALI_HOST_DEV void copy_assign(const SmallVector<T, other_static_size, alloc> &v) {
    const T *src = v.data();
    if (capacity() >= v.size()) {
      size_t i;
      T  *ptr = data();
      // overwrite common length
      for (i = 0; i < v.size() && i < size(); i++) {
        ptr[i] = src[i];
      }
      // construct new elements
      for (; i < v.size(); i++) {
        new(ptr+i) T(src[i]);
#ifndef __CUDA_ARCH__
        if (!noexcept(new(ptr+i) T(src[i])))
          set_size(i + 1);
#endif
      }
      // destroy tail, if any
      for (; i < size(); i++) {
        ptr[i].~T();
      }
      set_size(v.size());
    } else {
      clear();
      reserve(v.size());
      T *ptr = data();
      if (is_pod_v<T>) {
        set_size(v.size());
        for (size_t i = 0; i < size(); i++) {
          ptr[i] = src[i];
        }
      } else {
        for (size_t i = 0; i < v.size(); i++) {
          new(ptr + i) T(src[i]);
#ifndef __CUDA_ARCH__
          if (!noexcept(new(ptr + i) T(src[i])))
            set_size(i + 1);
#endif
        }
        set_size(v.size());
      }
    }
  }

  DALI_NO_EXEC_CHECK
  template <size_t other_static_size, typename alloc>
  DALI_HOST_DEV bool operator==(const SmallVector<T, other_static_size, alloc> &v) const {
    size_t n = size();
    if (n != v.size())
      return false;
    for (size_t i = 0; i < n; i++) {
      if ((*this)[i] != v[i])
        return false;
    }
    return true;
  }

  DALI_NO_EXEC_CHECK
  template <size_t other_static_size, typename alloc>
  DALI_HOST_DEV bool operator!=(const SmallVector<T, other_static_size, alloc> &v) const {
    return !(*this == v);
  }

  DALI_NO_EXEC_CHECK
  template <size_t other_static_size>
  DALI_HOST_DEV SmallVector &operator=(SmallVector<T, other_static_size, allocator> &&v) {
    if (v.is_dynamic() && v.capacity() > static_size) {
      clear();
      if (is_dynamic()) {
        deallocate(dynamic_data(), capacity());
      }
      set_dynamic_data(v.dynamic_data());
      set_size(v.size());
      set_capacity(v.capacity());
      v.set_dynamic_data(nullptr);
      v.set_size(0);
      v.reset_capacity();
    } else {
      clear();
      if (is_dynamic() && v.size() <= static_size) {
        deallocate(dynamic_data(), capacity());
        reset_capacity();
      } else {
        reserve(v.size());
      }
      T *src = v.data();
      T *dst = data();
      if (is_pod_v<T>) {
        this->copy(dst, src, v.size());
        set_size(v.size());
      } else {
        for (size_t i = 0; i < v.size(); i++) {
          new(dst + i) T(cuda_move(src[i]));
#ifndef __CUDA_ARCH__
          if (!noexcept(new(dst + i) T(cuda_move(src[i]))))
            set_size(i + 1);  // if the `new` above throws, we're in a consistent state
#endif
        }
        set_size(v.size());
      }
      v.clear();
    }
    return *this;
  }

  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using const_reference = const T&;
  using iterator = T*;
  using const_iterator = const T*;
  using index_type = std::ptrdiff_t;

  DALI_HOST_DEV inline iterator begin() {
    return data();
  }
  DALI_HOST_DEV inline iterator end() {
    return data() + size();
  }
  DALI_HOST_DEV inline const_iterator cbegin() const {
    return data();
  }
  DALI_HOST_DEV inline const_iterator cend() const {
    return data() + size();
  }

  DALI_HOST_DEV inline const_iterator begin() const {
    return cbegin();
  }
  DALI_HOST_DEV inline const_iterator end() const {
    return cend();
  }

  DALI_HOST_DEV inline reference front() {
    return data()[0];
  }
  DALI_HOST_DEV inline const_reference front() const {
    return data()[0];
  }

  DALI_HOST_DEV inline reference back() {
    return data()[size() - 1];
  }
  DALI_HOST_DEV inline const_reference back() const {
    return data()[size() - 1];
  }

  DALI_HOST_DEV inline size_t size() const {
    return size_ & size_mask;
  }

  DALI_HOST_DEV inline bool empty() const {
    return size() == 0;
  }

  DALI_HOST_DEV inline size_t capacity() const {
    return is_dynamic() ? dynamic.capacity : static_size;
  }

  DALI_HOST_DEV inline T *data() {
    if (is_dynamic()) {
      return dynamic_data();
    } else {
      return static_data();
    }
  }

  DALI_HOST_DEV inline const T *data() const {
    if (is_dynamic()) {
      return dynamic_data();
    } else {
      return static_data();
    }
  }

  DALI_HOST_DEV inline bool is_dynamic() const {
    return (size_ & dynamic_flag) != 0;
  }

  DALI_HOST_DEV inline reference operator[](ptrdiff_t index) {
    return data()[index];
  }

  DALI_HOST_DEV inline const_reference operator[](ptrdiff_t index) const {
    return data()[index];
  }

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV inline void clear() {
    this->destroy(data(), size());
    set_size(0);
  }

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV inline void push_back(const T &value) {
    emplace_back(value);
  }

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV inline void push_back(T &&value) {
    emplace_back(cuda_move(value));
  }

  DALI_NO_EXEC_CHECK
  template <typename... Args>
  DALI_HOST_DEV inline void emplace_back(Args&&... args) {
    pre_append();
    T *ptr = data();
    new(ptr + size_) T(cuda_forward<Args>(args)...);
    set_size(size() + 1);
  }

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV inline iterator insert(const_iterator before, const T &value) {
    return emplace(before, value);
  }

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV inline iterator insert(const_iterator before, T &&value) {
    return emplace(before, cuda_move(value));
  }

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV inline iterator insert_at(index_type index, const T &value) {
    return emplace_at(index, value);
  }

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV inline iterator insert_at(index_type index, T &&value) {
    return emplace_at(index, cuda_move(value));
  }

  DALI_NO_EXEC_CHECK
  template <typename... Args>
  DALI_HOST_DEV inline iterator emplace(const_iterator before, Args&&... args) {
    return emplace_at(before - begin(), cuda_forward<Args>(args)...);
  }

  DALI_NO_EXEC_CHECK
  template <typename... Args>
  DALI_HOST_DEV inline iterator emplace_at(index_type index, Args&&... args) {
    if (index == static_cast<index_type>(size())) {
      emplace_back(cuda_forward<Args>(args)...);
      return begin() + index;
    }
    T *ptr = data();

    if (size() + 1 > capacity()) {
      size_t new_capacity = cuda_max(size() + 1, 2 * capacity());
      bool dynamic;
      T *new_data = allocate(new_capacity, dynamic);
      index_type i = -1;
      index_type n = size();

      if (is_pod_v<T>) {
        for (i = 0; i < index; i++)
          new_data[i] = ptr[i];

        new_data[i] = T(cuda_forward<Args>(args)...);
        i++;
        for (; i <= n; i++)
          new_data[i] = ptr[i - 1];
      } else {
          new(new_data + index) T(cuda_forward<Args>(args)...);
          for (i = 0; i < index; i++)
            new(new_data + i) T(cuda_move(ptr[i]));
          i++;

          for (; i <= n; i++)
            new(new_data + i) T(cuda_move(ptr[i - 1]));
        this->destroy(ptr, n);
      }

      if (is_dynamic())
        deallocate(ptr, capacity());

      set_dynamic_data(new_data);
      set_capacity(new_capacity);
      set_size(size() + 1);
      return begin() + index;
    }
    make_space(index);
    ptr[index] = T(cuda_forward<Args>(args)...);
    set_size(size() + 1);
    return begin() + index;
  }

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV inline iterator erase(const_iterator position) {
    return erase_at(position - begin());
  }

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV inline iterator erase(const_iterator first, const_iterator last) {
    return erase_at(first - begin(), last - first);
  }

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV inline iterator erase_at(index_type first, size_t count = 1) {
    index_type n = size();
    T *ptr = data();

    for (index_type dst = first, src = first + count; src < n; dst++, src++)
      ptr[dst] = cuda_move(ptr[src]);

    if (!is_pod_v<T>) {
      for (index_type i = n - count; i < n; i++)
        ptr[i].~T();
    }

    set_size(n - count);
    return ptr + first;
  }

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV inline void pop_back() {
    back().~T();
    set_size(size() - 1);
  }

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV void resize(size_t count) {
    reserve(count);
    T *ptr = data();
    for (size_t i = size(); i < count; i++) {
      new(ptr + i) T();
    }
    for (size_t i = count; i < size(); i++) {
      ptr[i].~T();
    }
    set_size(count);
  }

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV void resize(size_t count, const value_type &value) {
    reserve(count);
    T *ptr = data();
    for (size_t i = size(); i < count; i++) {
      new(ptr + i) T(value);
    }
    for (size_t i = count; i < size(); i++) {
      ptr[i].~T();
    }
    set_size(count);
  }

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV void reserve(size_t new_capacity) {
    if (new_capacity <= capacity())
      return;
    T *ptr = data();
    bool dynamic;
    T *new_data = allocate(new_capacity, dynamic);
    this->move_and_destroy(new_data, ptr, size());
    if (is_dynamic())
      deallocate(ptr, capacity());
    set_dynamic_data(new_data);
    set_capacity(new_capacity);
  }

  DALI_NO_EXEC_CHECK
  DALI_HOST_DEV void shrink_to_fit() {
    size_t new_capacity = cuda_max(size_, static_size);
    if (capacity() > new_capacity) {
      T *ptr = data();
      bool dynamic;
      T *new_data = allocate(new_capacity, dynamic);
      for (size_t i = 0; i < size(); i++) {
        new(new_data + i) T(cuda_move(ptr[i]));
        ptr[i].~T();
      }
      deallocate(ptr);
      if (dynamic)
        set_dynamic_data(new_data);
      set_capacity(new_capacity);
    }
  }

  __host__ std::vector<T> to_vector() const {
    return std::vector<T>(begin(), end());
  }

 private:
  template <typename U, size_t n, typename A>
  friend class SmallVector;

  union {
    alignas(T) std::byte storage[sizeof(T) * static_size];  // NOLINT(runtime/arrays)
    struct {
      T *data;
      size_t capacity;
    } dynamic;
  };
  size_t size_ = 0;
  static constexpr size_t size_mask = -1_uz >> 1;
  static constexpr size_t dynamic_flag = ~size_mask;

  template <typename U>
  DALI_HOST_DEV inline void swap(U &a, U &b) {
    U tmp = cuda_move(a);
    a = cuda_move(b);
    b = cuda_move(tmp);
  }

  DALI_HOST_DEV inline void make_space(size_t index) {
    T *ptr = data();
    size_t n = size();
    size_t i = n;
    new(ptr + i)T(cuda_move(ptr[i-1]));
    for (; i > index; i--) {
      ptr[i] = cuda_move(ptr[i-1]);
    }
  }

  DALI_HOST_DEV inline void set_size(size_t new_size) {
    size_ = new_size | (size_ & dynamic_flag);
  }
  DALI_HOST_DEV inline void set_capacity(size_t new_capacity) {
    if (new_capacity > static_size) {
      size_ |= dynamic_flag;
      dynamic.capacity = new_capacity;
    } else {
      reset_capacity();
    }
  }
  DALI_HOST_DEV inline void reset_capacity() {
    size_ &= ~dynamic_flag;
  }

  DALI_HOST_DEV inline T *allocate(size_t count, bool &dynamic) {
    if (count <= static_size) {
      dynamic = false;
      return static_data();
    } else {
      dynamic = true;
      return Alloc::allocate(count);
    }
  }
  using Alloc::deallocate;

  DALI_HOST_DEV inline T *static_data() {
    return reinterpret_cast<T *>(storage);
  }
  DALI_HOST_DEV inline const T *static_data() const {
    return reinterpret_cast<const T *>(storage);
  }

  DALI_HOST_DEV inline T *dynamic_data() {
    return dynamic.data;
  }
  DALI_HOST_DEV inline const T *dynamic_data() const {
    return dynamic.data;
  }
  DALI_HOST_DEV inline void set_dynamic_data(T *ptr) {
    dynamic.data = ptr;
  }

  DALI_HOST_DEV inline void pre_append(size_t count = 1) {
    if (size() + count > capacity()) {
      reserve(cuda_max(2 * capacity(), size() + count));
    }
  }
};

}  // namespace dali

#pragma GCC diagnostic pop

#endif  // DALI_CORE_SMALL_VECTOR_H_
