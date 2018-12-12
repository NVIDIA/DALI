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

#ifndef DALI_KERNELS_SPAN_H_
#define DALI_KERNELS_SPAN_H_

#include <cstddef>
#include <type_traits>

namespace tensor {

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
  // using reverse_iterator = std::reverse_iterator<iterator>;
  // using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  static constexpr index_type extent = Extent;

  // [span.cons], span constructors, copy, assignment, and destructor
  // This constructor shall not participate in overload resolution unless Extent <= 0 is true
  constexpr span() noexcept = delete;
  constexpr span(pointer ptr, index_type count) : data_(ptr) { /* assert(count == Extent); */ }
  constexpr span(pointer firstElem, pointer lastElem) : data_(firstElem) {
    /* assert(lastElem - firstElem == Extent); */
  }

  // template <size_t N>
  // constexpr span(element_type (&arr)[N]) noexcept;
  // template <size_t N>
  // constexpr span(array<value_type, N> &arr) noexcept;
  // template <size_t N>
  // constexpr span(const array<value_type, N> &arr) noexcept;
  // template <class Container>
  // constexpr span(Container &&cont);
  // template <class Container>
  // constexpr span(const Container &cont);
  // constexpr span(const span &other) noexcept = default;
  // template <class OtherElementType, ptrdiff_t OtherExtent>
  // constexpr span(const span<OtherElementType, OtherExtent> &s) noexcept;
  ~span() noexcept = default;
  span &operator=(const span &other) noexcept = default;

  // [span.sub], span subviews
  // template <ptrdiff_t Count>
  // constexpr span<element_type, Count> first() const;
  // template <ptrdiff_t Count>
  // constexpr span<element_type, Count> last() const;
  // template <ptrdiff_t Offset, ptrdiff_t Count = dynamic_extent>
  // constexpr span<element_type, /* see below */> subspan() const;
  // constexpr span<element_type, dynamic_extent> first(index_type count) const;
  // constexpr span<element_type, dynamic_extent> last(index_type count) const;
  // constexpr span<element_type, dynamic_extent> subspan(index_type offset,
  //                                                      index_type count = dynamic_extent) const;

  // [span.obs], span observers
  constexpr index_type size() const noexcept { return Extent; }
  constexpr index_type size_bytes() const noexcept { return Extent * sizeof(value_type); }
  constexpr bool empty() const noexcept { return false; }

  // [span.elem], span element access
  constexpr reference operator[](index_type idx) const { return data_[idx]; }
  // constexpr reference operator()(index_type idx) const;
  constexpr pointer data() const noexcept { return data_; }

  // [span.iterators], span iterator support
  constexpr iterator begin() const noexcept { return data_; }
  constexpr iterator end() const noexcept { return data_ + Extent; }
  constexpr const_iterator cbegin() const noexcept { return data_; }
  constexpr const_iterator cend() const noexcept { return data_ + Extent; }
  // constexpr reverse_iterator rbegin() const noexcept;
  // constexpr reverse_iterator rend() const noexcept;
  // constexpr const_reverse_iterator crbegin() const noexcept;
  // constexpr const_reverse_iterator crend() const noexcept;

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
  // using reverse_iterator = std::reverse_iterator<iterator>;
  // using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  static constexpr index_type extent = dynamic_extent;

  // [span.cons], span constructors, copy, assignment, and destructor
  constexpr span() noexcept : data_(nullptr), size_(0) {}
  constexpr span(pointer ptr, index_type count) : data_(ptr), size_(count) {}
  constexpr span(pointer firstElem, pointer lastElem) : data_(firstElem), size_(lastElem - firstElem) {}
  // template <size_t N>
  // constexpr span(element_type (&arr)[N]) noexcept;
  // template <size_t N>
  // constexpr span(array<value_type, N> &arr) noexcept;
  // template <size_t N>
  // constexpr span(const array<value_type, N> &arr) noexcept;
  // template <class Container>
  // constexpr span(Container &cont);
  // template <class Container>
  // constexpr span(const Container &cont);
  constexpr span(const span &other) noexcept = default;
  // template <class OtherElementType, ptrdiff_t OtherExtent>
  // constexpr span(const span<OtherElementType, OtherExtent> &s) noexcept;
  ~span() noexcept = default;
  span &operator=(const span &other) noexcept = default;

  // [span.sub], span subviews
  // template <ptrdiff_t Count>
  // constexpr span<element_type, Count> first() const;
  // template <ptrdiff_t Count>
  // constexpr span<element_type, Count> last() const;
  // template <ptrdiff_t Offset, ptrdiff_t Count = dynamic_extent>
  // constexpr span<element_type, /* see below */> subspan() const;
  // constexpr span<element_type, dynamic_extent> first(index_type count) const;
  // constexpr span<element_type, dynamic_extent> last(index_type count) const;
  // constexpr span<element_type, dynamic_extent> subspan(index_type offset,
  //                                                      index_type count = dynamic_extent) const;

  // [span.obs], span observers
  constexpr index_type size() const noexcept { return size_; }
  constexpr index_type size_bytes() const noexcept { return size_ * sizeof(value_type); }
  constexpr bool empty() const noexcept { return size() == 0; }

  // [span.elem], span element access
  constexpr reference operator[](index_type idx) const { return data_[idx]; }
  // constexpr reference operator()(index_type idx) const;
  constexpr pointer data() const noexcept { return data_; }

  // [span.iterators], span iterator support
  constexpr iterator begin() const noexcept { return data_; }
  constexpr iterator end() const noexcept { return data_ + size(); }
  constexpr const_iterator cbegin() const noexcept { return data_; }
  constexpr const_iterator cend() const noexcept { return data_ + size(); }
  // constexpr reverse_iterator rbegin() const noexcept;
  // constexpr reverse_iterator rend() const noexcept;
  // constexpr const_reverse_iterator crbegin() const noexcept;
  // constexpr const_reverse_iterator crend() const noexcept;

 private:
  pointer data_;
  index_type size_;
};

}  // namespace tensor

#endif  // DALI_KERNELS_SPAN_H_
