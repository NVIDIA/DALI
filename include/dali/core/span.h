// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_CORE_SPAN_H_
#define DALI_CORE_SPAN_H_

#include <cstddef>
#include <array>
#include <type_traits>
#include "dali/core/host_dev.h"

namespace dali {

// Based on "span: bounds-safe views for sequences of objects"
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0122r7.pdf
// and "Usability Enhancements for std::span"
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1024r3.pdf
// Adopted to c++20
// Missing: containers constructors, static <-> dynamic, reverse iterators, subspans, comparisions

using span_extent_t = ptrdiff_t;
constexpr span_extent_t dynamic_extent = span_extent_t(-1);

template <class ElementType, span_extent_t Extent = dynamic_extent>
class span {
 public:
  // constants and types
  using element_type = ElementType;
  using value_type = std::remove_cv_t<ElementType>;
  using index_type = span_extent_t;
  using difference_type = ptrdiff_t;
  using pointer = element_type *;
  using reference = element_type &;
  using iterator = element_type *;
  using const_iterator = const element_type *;
  static constexpr index_type extent = Extent;

  // [span.cons], span constructors, copy, assignment, and destructor
  // This constructor shall not participate in overload resolution unless Extent <= 0 is true
  constexpr span() noexcept = delete;
  DALI_HOST_DEV constexpr span(pointer ptr, index_type count = Extent) : data_(ptr) {  // NOLINT
    /* assert(count == Extent); */
  }

  template <class U, typename = std::enable_if_t<
    std::is_convertible<U(*)[], ElementType(*)[]>::value
  >>
  DALI_HOST_DEV constexpr span(const span<U, Extent>& s) noexcept : data_(s.data()) {}

  DALI_HOST_DEV constexpr span(pointer firstElem, pointer lastElem) : data_(firstElem) {
    /* assert(lastElem - firstElem == Extent); */
  }

  ~span() noexcept = default;
  span &operator=(const span &other) noexcept = default;

  // [span.sub], span subviews

  // [span.obs], span observers
  DALI_HOST_DEV constexpr index_type size() const noexcept { return Extent; }
  DALI_HOST_DEV constexpr index_type size_bytes() const noexcept {
    return Extent * sizeof(value_type);
  }
  DALI_HOST_DEV constexpr bool empty() const noexcept { return false; }

  // [span.elem], span element access
  DALI_HOST_DEV constexpr reference operator[](index_type idx) const { return data_[idx]; }
  DALI_HOST_DEV constexpr pointer data() const noexcept { return data_; }

  // [span.iterators], span iterator support
  DALI_HOST_DEV constexpr iterator begin() const noexcept { return data_; }
  DALI_HOST_DEV constexpr iterator end() const noexcept { return data_ + Extent; }
  DALI_HOST_DEV constexpr const_iterator cbegin() const noexcept { return data_; }
  DALI_HOST_DEV constexpr const_iterator cend() const noexcept { return data_ + Extent; }

  DALI_HOST_DEV constexpr reference front() const noexcept { return data_[0]; }
  DALI_HOST_DEV constexpr reference back() const noexcept { return data_[Extent - 1]; }

 private:
  pointer data_;
};

template <class ElementType>
class span<ElementType, dynamic_extent> {
 public:
  // constants and types
  using element_type = ElementType;
  using value_type = std::remove_cv_t<ElementType>;
  using index_type = span_extent_t;
  using difference_type = ptrdiff_t;
  using pointer = element_type *;
  using reference = element_type &;
  using iterator = element_type *;
  using const_iterator = const element_type *;
  static constexpr index_type extent = dynamic_extent;

  // [span.cons], span constructors, copy, assignment, and destructor
  DALI_HOST_DEV constexpr span() noexcept : data_(nullptr), size_(0) {}
  DALI_HOST_DEV constexpr span(pointer ptr, index_type count) : data_(ptr), size_(count) {}
  DALI_HOST_DEV constexpr span(pointer firstElem, pointer lastElem)
      : data_(firstElem), size_(lastElem - firstElem) {}

  template <class U, span_extent_t N, typename = std::enable_if_t<
    std::is_convertible<U(*)[], ElementType(*)[]>::value
  >>
  DALI_HOST_DEV constexpr span(const span<U, N>& s) noexcept : span(s.data(), s.size()) {}

  constexpr span(const span &other) noexcept = default;
  ~span() noexcept = default;
  span &operator=(const span &other) noexcept = default;

  // [span.sub], span subviews

  // [span.obs], span observers
  DALI_HOST_DEV constexpr index_type size() const noexcept { return size_; }
  DALI_HOST_DEV constexpr index_type size_bytes() const noexcept {
    return size_ * sizeof(value_type);
  }
  DALI_HOST_DEV constexpr bool empty() const noexcept { return size() == 0; }

  // [span.elem], span element access
  DALI_HOST_DEV constexpr reference operator[](index_type idx) const { return data_[idx]; }
  DALI_HOST_DEV constexpr pointer data() const noexcept { return data_; }

  // [span.iterators], span iterator support
  DALI_HOST_DEV constexpr iterator begin() const noexcept { return data_; }
  DALI_HOST_DEV constexpr iterator end() const noexcept { return data_ + size(); }
  DALI_HOST_DEV constexpr const_iterator cbegin() const noexcept { return data_; }
  DALI_HOST_DEV constexpr const_iterator cend() const noexcept { return data_ + size(); }

  DALI_HOST_DEV constexpr reference front() const noexcept { return data_[0]; }
  DALI_HOST_DEV constexpr reference back() const noexcept { return data_[size_ - 1]; }

 private:
  pointer data_;
  index_type size_;
};

// [span.comparison], span comparison operators
template <class ElementL, span_extent_t ExtentL, class ElementR, span_extent_t ExtentR>
DALI_HOST_DEV constexpr bool
operator==(span<ElementL, ExtentL> l, span<ElementR, ExtentR> r) {
  if (l.size() != r.size()) {
    return false;
  }
  auto size = l.size();
  for (decltype(size) i = 0; i < size; i++) {
    if (l[i] != r[i]) {
      return false;
    }
  }
  return true;
}
template <class ElementL, span_extent_t ExtentL, class ElementR, span_extent_t ExtentR>
DALI_HOST_DEV constexpr bool
operator!=(span<ElementL, ExtentL> l, span<ElementR, ExtentR> r) {
  return !(l == r);
}

// @brief Helper function for pre-C++17
template <span_extent_t Extent, typename T>
DALI_HOST_DEV constexpr span<T, Extent> make_span(T *data) { return { data }; }

// @brief Helper function for pre-C++17
template <span_extent_t Extent = dynamic_extent, typename T>
DALI_HOST_DEV constexpr span<T, Extent> make_span(T *data, span_extent_t extent) {
  return { data, extent };
}

DALI_NO_EXEC_CHECK
template <typename Collection>
DALI_HOST_DEV constexpr auto make_span(Collection &c) {
  return make_span(c.data(), c.size());
}

DALI_NO_EXEC_CHECK
template <typename Collection>
DALI_HOST_DEV constexpr auto make_span(Collection &&c) {
  static_assert(!std::is_rvalue_reference<Collection&&>::value,
    "Cannot create a span from an r-value.");
  return make_span(c.data(), c.size());
}

DALI_NO_EXEC_CHECK
template <typename T, size_t N>
DALI_HOST_DEV constexpr span<T, N> make_span(std::array<T, N> &a) {
  return { a.data() };
}

DALI_NO_EXEC_CHECK
template <typename T, size_t N>
DALI_HOST_DEV constexpr span<const T, N> make_span(const std::array<T, N> &a) {
  return { a.data() };
}

DALI_NO_EXEC_CHECK
template <typename T, size_t N>
DALI_HOST_DEV constexpr span<const T, N> make_span(std::array<T, N> &&a) {
  static_assert(!std::is_rvalue_reference<std::array<T, N> &&>::value,
    "Cannot create a span from an r-value.");
  return { a.data() };
}

template <typename T, size_t N>
DALI_HOST_DEV constexpr span<T, N> make_span(T (&a)[N]) {
  return { a };
}

template <span_extent_t Extent, typename T>
DALI_HOST_DEV constexpr span<const T, Extent> make_cspan(T *data) { return { data }; }

template <span_extent_t Extent = dynamic_extent, typename T>
DALI_HOST_DEV constexpr span<const T, Extent> make_cspan(const T *data, span_extent_t extent) {
  return { data, extent };
}

DALI_NO_EXEC_CHECK
template <typename Collection>
DALI_HOST_DEV constexpr auto make_cspan(Collection &c) {
  return make_cspan(c.data(), c.size());
}

DALI_NO_EXEC_CHECK
template <typename Collection>
DALI_HOST_DEV constexpr auto make_cspan(Collection &&c) {
  static_assert(!std::is_rvalue_reference<Collection&&>::value,
                "Cannot create a span from an r-value.");
  return make_cspan(c.data(), c.size());
}

DALI_NO_EXEC_CHECK
template <typename T, size_t N>
DALI_HOST_DEV constexpr span<const T, N> make_cspan(const std::array<T, N> &a) {
  return { a.data() };
}

DALI_NO_EXEC_CHECK
template <typename T, size_t N>
DALI_HOST_DEV constexpr span<const T, N> make_cspan(std::array<T, N> &&a) {
  static_assert(!std::is_rvalue_reference<std::array<T, N> &&>::value,
                "Cannot create a span from an r-value.");
  return { a.data() };
}

template <typename T, size_t N>
DALI_HOST_DEV constexpr span<const T, N> make_cspan(T (&a)[N]) {
  return { a };
}

}  // namespace dali

#endif  // DALI_CORE_SPAN_H_
