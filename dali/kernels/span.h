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

#ifndef DALI_KERNELS_SPAN_H_
#define DALI_KERNELS_SPAN_H_

#include <cstddef>
#include <array>
#include <type_traits>

namespace dali {

// Based on "span: bounds-safe views for sequences of objects"
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0122r7.pdf
// Adopted to c++20
// Missing: containers constructors, static <-> dynamic, reverse iterators, subspans, comparisions
// Removed some constexprs due to c++11 limitations

constexpr ptrdiff_t dynamic_extent = -1;

template <class ElementType, ptrdiff_t Extent = dynamic_extent>
class span {
 public:
  // constants and types
  using element_type = ElementType;
  using value_type = typename std::remove_cv<ElementType>::type;
  using index_type = ptrdiff_t;
  using difference_type = ptrdiff_t;
  using pointer = element_type *;
  using reference = element_type &;
  using iterator = element_type *;
  using const_iterator = const element_type *;
  static constexpr index_type extent = Extent;

  // [span.cons], span constructors, copy, assignment, and destructor
  // This constructor shall not participate in overload resolution unless Extent <= 0 is true
  constexpr span() noexcept = delete;
  constexpr span(pointer ptr, index_type count = Extent) : data_(ptr) {  // NOLINT
    /* assert(count == Extent); */
  }

  constexpr span(pointer firstElem, pointer lastElem) : data_(firstElem) {
    /* assert(lastElem - firstElem == Extent); */
  }

  ~span() noexcept = default;
  span &operator=(const span &other) noexcept = default;

  // [span.sub], span subviews

  // [span.obs], span observers
  constexpr index_type size() const noexcept { return Extent; }
  constexpr index_type size_bytes() const noexcept { return Extent * sizeof(value_type); }
  constexpr bool empty() const noexcept { return false; }

  // [span.elem], span element access
  constexpr reference operator[](index_type idx) const { return data_[idx]; }
  constexpr reference operator()(index_type idx) const { return data_[idx]; }
  constexpr pointer data() const noexcept { return data_; }

  // [span.iterators], span iterator support
  constexpr iterator begin() const noexcept { return data_; }
  constexpr iterator end() const noexcept { return data_ + Extent; }
  constexpr const_iterator cbegin() const noexcept { return data_; }
  constexpr const_iterator cend() const noexcept { return data_ + Extent; }

 private:
  pointer data_;
};

template <class ElementType>
class span<ElementType, dynamic_extent> {
 public:
  // constants and types
  using element_type = ElementType;
  using value_type = typename std::remove_cv<ElementType>::type;
  using index_type = ptrdiff_t;
  using difference_type = ptrdiff_t;
  using pointer = element_type *;
  using reference = element_type &;
  using iterator = element_type *;
  using const_iterator = const element_type *;
  static constexpr index_type extent = dynamic_extent;

  // [span.cons], span constructors, copy, assignment, and destructor
  constexpr span() noexcept : data_(nullptr), size_(0) {}
  constexpr span(pointer ptr, index_type count) : data_(ptr), size_(count) {}
  constexpr span(pointer firstElem, pointer lastElem)
      : data_(firstElem), size_(lastElem - firstElem) {}

  constexpr span(const span &other) noexcept = default;
  ~span() noexcept = default;
  span &operator=(const span &other) noexcept = default;

  // [span.sub], span subviews

  // [span.obs], span observers
  constexpr index_type size() const noexcept { return size_; }
  constexpr index_type size_bytes() const noexcept { return size_ * sizeof(value_type); }
  constexpr bool empty() const noexcept { return size() == 0; }

  // [span.elem], span element access
  constexpr reference operator[](index_type idx) const { return data_[idx]; }
  constexpr reference operator()(index_type idx) const { return data_[idx]; }
  constexpr pointer data() const noexcept { return data_; }

  // [span.iterators], span iterator support
  constexpr iterator begin() const noexcept { return data_; }
  constexpr iterator end() const noexcept { return data_ + size(); }
  constexpr const_iterator cbegin() const noexcept { return data_; }
  constexpr const_iterator cend() const noexcept { return data_ + size(); }

 private:
  pointer data_;
  index_type size_;
};

// [span.comparison], span comparison operators
template <class ElementL, ptrdiff_t ExtentL, class ElementR, ptrdiff_t ExtentR>
/* constexpr */ bool operator==(span<ElementL, ExtentL> l, span<ElementR, ExtentR> r) {
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
template <class ElementL, ptrdiff_t ExtentL, class ElementR, ptrdiff_t ExtentR>
/* constexpr */ bool operator!=(span<ElementL, ExtentL> l, span<ElementR, ExtentR> r) {
  return !(l == r);
}

// @brief Helper function for pre-C++17
template <ptrdiff_t Extent, typename T>
constexpr span<T, Extent> make_span(T *data) { return { data }; }

// @brief Helper function for pre-C++17
template <ptrdiff_t Extent = dynamic_extent, typename T>
constexpr span<T, Extent> make_span(T *data, ptrdiff_t extent) { return { data, extent }; }

template <typename Collection>
auto make_span(Collection &c)->decltype(make_span(c.data(), c.size())) {
  return make_span(c.data(), c.size());
}

template <typename Collection>
auto make_span(Collection &&c)->decltype(make_span(c.data(), c.size())) {
  static_assert(!std::is_rvalue_reference<Collection&&>::value,
    "Cannot create a span from an r-value.");
  return make_span(c.data(), c.size());
}

template <typename T, size_t N>
constexpr span<T, N> make_span(std::array<T, N> &a) {
  return { a.data() };
}

template <typename T, size_t N>
constexpr span<const T, N> make_span(const std::array<T, N> &a) {
  return { a.data() };
}

template <typename T, size_t N>
constexpr span<const T, N> make_span(std::array<T, N> &&a) {
  static_assert(!std::is_rvalue_reference<std::array<T, N> &&>::value,
    "Cannot create a span from an r-value.");
  return { a.data() };
}

template <typename T, size_t N>
constexpr span<const T, N> make_span(T (&a)[N]) {
  return { a };
}

}  // namespace dali

#endif  // DALI_KERNELS_SPAN_H_
