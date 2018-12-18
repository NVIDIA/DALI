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

#ifndef DALI_KERNELS_SHAPE_H_
#define DALI_KERNELS_SHAPE_H_

#include <array>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

#include "dali/kernels/span.h"

namespace dali {
namespace kernels {



/// @brief Returns the product of all elements in shape
/// @param shape - shape of a tensor whose elements we count
template <typename T>
inline int64_t volume(const T &shape) {
  int n = shape.size();
  if (n < 1) {
    return 0;
  }
  int64_t v = shape[0];
  for (int i = 1; i < n; i++) {
    v *= shape[i];
  }
  return v;
}

constexpr int DynamicDimensions = -1;

/// @brief Class representing shape of a Tensor
///
/// Static shapes do not allocate additional memory as they are backed by std::array
/// @tparam ndim Either non-negative integer representing static number of dimensions
///         or DynamicDimensions.
template <int ndim = DynamicDimensions>
struct TensorShape;

/// @brief Base class for TensorShape containing common code for iterators and operator[]
template <typename Container, int ndim>
struct TensorShapeBase {
  using container_type = Container;
  using value_type = typename container_type::value_type;
  using size_type = int;
  using reference = value_type &;
  using const_reference = const value_type &;
  using iterator = typename container_type::iterator;
  using const_iterator = typename container_type::const_iterator;
  using reverse_iterator = typename container_type::reverse_iterator;
  using const_reverse_iterator = typename container_type::const_reverse_iterator;

  reference operator[](int d) { return shape[d]; }
  const_reference operator[](int d) const { return shape[d]; }

  iterator begin() noexcept { return shape.begin(); }
  iterator end() noexcept { return shape.end(); }
  const_iterator begin() const noexcept { return shape.begin(); }
  const_iterator end() const noexcept { return shape.end(); }
  const_iterator cbegin() const noexcept { return shape.cbegin(); }
  const_iterator cend() const noexcept { return shape.cend(); }
  reverse_iterator rbegin() noexcept { return shape.rbegin(); }
  reverse_iterator rend() noexcept { return shape.rend(); }
  const_reverse_iterator rbegin() const noexcept { return shape.rbegin(); }
  const_reverse_iterator rend() const noexcept { return shape.rend(); }
  const_reverse_iterator crbegin() const noexcept { return shape.crbegin(); }
  const_reverse_iterator crend() const noexcept { return shape.crend(); }

  /// @brief Returns number of dimensions in this shape
  size_type size() const { return shape.size(); }
  /// @brief Returns number of dimensions in this shape
  size_type sample_dim() const { return shape.size(); }
  constexpr bool empty() const { return size() == 0; }

  Container shape;
  static constexpr int static_ndim = ndim;

  /// @brief Returns a static subshape consisting of first other_ndim dimensions (outer dimensions)
  /// [1, 2, 3, 4].first<2>() -> [1, 2]
  template <int other_ndim>
  TensorShape<other_ndim> first();
  /// @brief Returns a static subshape consisting of last other_ndim dimensions (inner dimensions)
  /// [1, 2, 3, 4].last<2>() -> [3, 4]
  template <int other_ndim>
  TensorShape<other_ndim> last();

  /// @brief Returns a dynamic subshape consisting of first count dimensions (outer dimensions)
  /// [1, 2, 3, 4].first(2) -> [1, 2]
  TensorShape<DynamicDimensions> first(int count);
  /// @brief Returns a dynamic subshape consisting of last count dimensions (inner dimensions)
  /// [1, 2, 3, 4].last(2) -> [3, 4]
  TensorShape<DynamicDimensions> last(int count);

 protected:
  // Disallow instatiation of Base class

  // Zero-fill the shape for Container=std::array<int64_t> with shape{}
  TensorShapeBase() : shape{} {}

  TensorShapeBase(const Container &c) : shape(c) {}
  TensorShapeBase(Container &&c) : shape(std::move(c)) {}
};

template <>
struct TensorShape<DynamicDimensions>
    : public TensorShapeBase<std::vector<int64_t>, DynamicDimensions> {
  using Base = TensorShapeBase<std::vector<int64_t>, DynamicDimensions>;
  TensorShape(const std::vector<int64_t> &s) : Base(s) {}

  template <size_t N>
  TensorShape(const std::array<int64_t, N> &s) : Base(typename Base::container_type(s.begin(), s.end())) {}

  template <typename... Ts>
  TensorShape(int64_t i0, Ts... s) : Base(typename Base::container_type{i0, int64_t{s}...}) {}

  TensorShape() = default;
  TensorShape(const TensorShape &) = default;
  TensorShape(TensorShape &&) = default;
  TensorShape &operator=(const TensorShape &other) = default;
  TensorShape &operator=(TensorShape &&other) = default;

  template <int other_ndim>
  TensorShape(const TensorShape<other_ndim> &other)
      : Base(typename Base::container_type(other.shape.begin(), other.shape.end())) {}

  template <int other_ndim>
  TensorShape &operator=(const TensorShape<other_ndim> &other) {
    shape = Base::container_type(other.shape.begin(), other.shape.end());
    return *this;
  }

  /// @brief Convert to static shape
  /// Behaviour is undefined for other_ndim != dim()
  template <int other_ndim>
  TensorShape<other_ndim> to_static() const {
    static_assert(other_ndim != DynamicDimensions,
                  "Conversion to static only allowed for static shape");
    assert(size() == other_ndim);
    TensorShape<other_ndim> shape;
    for (int i = 0; i < other_ndim; i++) {
      shape[i] = (*this)[i];
    }
    return shape;
  }

  void resize(typename Base::size_type count) { shape.resize(count); }
};

template <int ndim>
struct TensorShape : public TensorShapeBase<std::array<int64_t, ndim>, ndim> {
  using Base = TensorShapeBase<std::array<int64_t, ndim>, ndim>;
  TensorShape(const std::array<int64_t, ndim> &s) : Base(s) {}
  // Base class constructor will zero-initialize array
  TensorShape() = default;
  // We allow only explicit operations on TensorShape static dim
  TensorShape(const TensorShape &) = default;
  TensorShape &operator=(const TensorShape &other) = default;

  template <typename... Ts>
  TensorShape(int64_t i0, Ts... s) : Base(typename Base::container_type{i0, int64_t{s}...}) {
    static_assert(sizeof...(Ts) == ndim - 1, "Number of shapes passed must match ndim");
  }

  template <int other_ndim>
  TensorShape<other_ndim> to_static() const {
    static_assert(other_ndim != ndim, "Cannot convert to other static ndim");
    return *this;
  }

  void resize(typename Base::size_type count) {
    assert(count == ndim && "Not supported for count other than statically defined");
  }

  static_assert(ndim >= 0, "TensorShape dimension should not be negative");
};

template <typename Container, int ndim>
template <int other_ndim>
TensorShape<other_ndim> TensorShapeBase<Container, ndim>::first() {
  static_assert(other_ndim <= ndim || ndim == DynamicDimensions,
                "Number of dimensions in subshape must be between 0 and size()");
  static_assert(other_ndim != DynamicDimensions, "This function can produce only static shapes");
  assert(0 <= other_ndim && other_ndim <= size() &&
         "Number of dimensions in subshape must be between 0 and size()");
  TensorShape<other_ndim> result;
  for (int i = 0; i < other_ndim; i++) {
    result[i] = (*this)[i];
  }
  return result;
}

template <typename Container, int ndim>
template <int other_ndim>
TensorShape<other_ndim> TensorShapeBase<Container, ndim>::last() {
  static_assert(other_ndim <= ndim || ndim == DynamicDimensions,
                "Number of dimensions in subshape must be between 0 and size()");
  static_assert(other_ndim != DynamicDimensions, "This function can produce only static shapes");
  assert(0 <= other_ndim && other_ndim <= size() &&
         "Number of dimensions in subshape must be between 0 and size()");
  TensorShape<other_ndim> result;
  int start_offset = size() - other_ndim;
  for (int i = 0; i < other_ndim; i++) {
    result[i] = (*this)[start_offset + i];
  }
  return result;
}

template <typename Container, int ndim>
TensorShape<DynamicDimensions> TensorShapeBase<Container, ndim>::first(int count) {
  assert(0 <= count && count <= size() &&
         "Number of dimensions in subshape must be between 0 and size()");
  TensorShape<DynamicDimensions> result;
  result.resize(count);
  for (int i = 0; i < count; i++) {
    result[i] = (*this)[i];
  }
  return result;
}

template <typename Container, int ndim>
TensorShape<DynamicDimensions> TensorShapeBase<Container, ndim>::last(int count) {
  assert(0 <= count && count <= size() &&
         "Number of dimensions in subshape must be between 0 and size()");
  TensorShape<DynamicDimensions> result;
  result.resize(count);
  int start_offset = size() - count;
  for (int i = 0; i < count; i++) {
    result[i] = (*this)[start_offset + i];
  }
  return result;
}

/// @brief Checks if both shapes have the same number of dimensions and all of them are equal
template <int left_ndim, int right_ndim>
bool operator==(const TensorShape<left_ndim> &left, const TensorShape<right_ndim> &right) {
  if (left.size() != right.size()) {
    return false;
  }
  int size = left.size();
  for (int i = 0; i < size; i++) {
    if (left[i] != right[i]) {
      return false;
    }
  }
  return true;
}

template <int left_ndim, int right_ndim>
bool operator!=(const TensorShape<left_ndim> &left, const TensorShape<right_ndim> &right) {
  return !(left == right);
}

constexpr int shape_cat_ndim(int left_ndim, int right_ndim) {
  return (left_ndim == DynamicDimensions || right_ndim == DynamicDimensions)
             ? DynamicDimensions
             : left_ndim + right_ndim;
}

/// @brief Concatenate shapes
/// @return TensorShape<shape_cat_ndim(left_ndim, right_ndim)> Static shape if both of arguments
///         are static, otherwise dynamic
template <int left_ndim, int right_ndim>
TensorShape<shape_cat_ndim(left_ndim, right_ndim)> shape_cat(const TensorShape<left_ndim> &left,
                                                             const TensorShape<right_ndim> &right) {
  TensorShape<shape_cat_ndim(left_ndim, right_ndim)> result;
  int total_size = left.size() + right.size();
  result.resize(total_size);
  for (int i = 0; i < left.size(); i++) {
    result[i] = left[i];
  }
  for (int i = 0; i < right.size(); i++) {
    result[left.size() + i] = right[i];
  }
  return result;
}

/// @brief Flatten list of shapes into contigous vector
template <int sample_ndim>
typename std::enable_if<sample_ndim != DynamicDimensions, std::vector<int64_t>>::type
flatten_shapes(const std::vector<TensorShape<sample_ndim>> &shapes) {
  std::vector<int64_t> result;
  result.resize(sample_ndim * shapes.size());
  for (size_t sample = 0; sample < shapes.size(); sample++) {
    for (int axis = 0; axis < sample_ndim; axis++) {
      result[sample * sample_ndim + axis] = shapes[sample][axis];
    }
  }
  return result;
}

/// @brief Get the dim from list of shapes that have uniform dimensions.
/// @return 0 if list is empty, otherwise dim of first element
std::is_same<T, std::vector<int64_t>>::value,
int>::type
template <typename T>
typename std::enable_if<std::is_same<T, TensorShape<DynamicDimensions>>::value ||
                            std::is_same<T, std::vector<int64_t>>::value,
                        int>::type
get_dim_from_uniform(const std::vector<T> &shapes) {
  if (shapes.empty()) {
    return 0;
  }
  return shapes[0].size();
}

template <typename T>
typename std::enable_if<std::is_same<T, TensorShape<DynamicDimensions>>::value ||
                            std::is_same<T, std::vector<int64_t>>::value,
                        std::vector<int64_t>>::type
flatten_shapes(const std::vector<T> &shapes) {
  std::vector<int64_t> result;
  int uniform_sample_ndim = get_dim_from_uniform(shapes);
  result.resize(uniform_sample_ndim * shapes.size());
  for (size_t sample = 0; sample < shapes.size(); sample++) {
    assert(shapes[sample].size() == uniform_sample_ndim);
    for (int axis = 0; axis < uniform_sample_ndim; axis++) {
      result[sample * uniform_sample_ndim + axis] = shapes[sample][axis];
    }
  }
  return result;
}

/// @brief List of TensorShapes stored as contigous vector.
///        All shapes have the same number of dimensions
///
/// @tparam sample_ndim Either non-negative integer representing static number of dimensions
///         or DynamicDimensions.
template <int sample_ndim = DynamicDimensions>
struct TensorListShape;

template <typename Derived, int sample_ndim>
struct TensorListShapeBase {
  /// @brief Returns a static subshape list consisting of first other_ndim dimensions
  ///        (outer dimensions) for each sample
  template <int other_ndim>
  TensorListShape<other_ndim> first();
  /// @brief Returns a static subshape list consisting of last other_ndim dimensions
  ///        (inner dimensions) for each sample
  template <int other_ndim>
  TensorListShape<other_ndim> last();

  /// @brief Returns a dynamic subshape list consisting of first count dimensions
  ///        (outer dimensions) for each sample
  TensorListShape<DynamicDimensions> first(int count);
  /// @brief Returns a dynamic subshape list consisting of last count dimensions
  ///        (inner dimensions) for each sample
  TensorListShape<DynamicDimensions> last(int count);

  /// @brief Return a span containing the shape of `sample`
  span<int64_t, sample_ndim == DynamicDimensions ? dynamic_extent : sample_ndim> tensor_shape_span(
      int64_t sample) {
    return {&shapes[sample * get_sample_dim()], get_sample_dim()};
  }

  span<const int64_t, sample_ndim == DynamicDimensions ? dynamic_extent : sample_ndim>
  tensor_shape_span(int64_t sample) const {
    return {&shapes[sample * get_sample_dim()], get_sample_dim()};
  }

  /// @brief Return the TensorShape for given `sample`
  ///
  /// @tparam tensor_ndim Should be equal sample_dim() or DynamicDimensions to obtain either static
  ///         or dynamic TensorShape
  template <int tensor_ndim = sample_ndim>
  TensorShape<tensor_ndim> tensor_shape(int64_t sample) const {
    static_assert(tensor_ndim == sample_ndim || sample_ndim == DynamicDimensions,
                  "Cannot convert to other static ndim");
    assert(tensor_ndim == get_sample_dim() && "Cannot convert to other ndim");
    TensorShape<tensor_ndim> out;
    out.resize(get_sample_dim());
    int64_t base = get_sample_dim() * sample;
    for (int i = 0; i < get_sample_dim(); i++) {
      out[i] = shapes[base + i];
    }
    return out;
  }

  std::vector<int64_t> shapes;

  decltype(shapes.data()) data() {
    return shapes.data();
  }
  constexpr bool empty() const { return get_size() == 0; }
  int num_samples() const { return get_size(); }

 protected:
  int get_size() const { return static_cast<const Derived *>(this)->size(); }
  int get_sample_dim() const { return static_cast<const Derived *>(this)->sample_dim(); }
  TensorListShapeBase() = default;
  TensorListShapeBase(const std::vector<int64_t> &shapes) : shapes(shapes) {}
  TensorListShapeBase(std::vector<int64_t> &&shapes) : shapes(std::move(shapes)) {}
};

template <>
struct TensorListShape<DynamicDimensions>
    : TensorListShapeBase<TensorListShape<DynamicDimensions>, DynamicDimensions> {
  using Base = TensorListShapeBase<TensorListShape<DynamicDimensions>, DynamicDimensions>;

  TensorListShape() : Base(), dim(0) {}

  TensorListShape(const TensorListShape &) = default;
  TensorListShape(TensorListShape &&) = default;

  template <int other_sample_ndim>
  TensorListShape(const TensorListShape<other_sample_ndim> &other)
      : Base(other.shapes), dim(other_sample_ndim) {}

  template <int other_sample_ndim>
  TensorListShape(TensorListShape<other_sample_ndim> &&other)
      : Base(std::move(other.shapes)), dim(other_sample_ndim) {}

  TensorListShape(const std::vector<std::vector<int64_t>> &sample_shapes)
      : Base(flatten_shapes(sample_shapes)), dim(get_dim_from_uniform(sample_shapes)) {}

  TensorListShape(const std::vector<TensorShape<DynamicDimensions>> &sample_shapes)
      : Base(flatten_shapes(sample_shapes)), dim(get_dim_from_uniform(sample_shapes)) {}

  TensorListShape(const std::vector<int64_t> &shapes, int ndim) : Base(shapes), dim(ndim) {}
  TensorListShape(std::vector<int64_t> &&shapes, int ndim) : Base(std::move(shapes)), dim(ndim) {}

  /// @brief Return a dynamic TensorShape for `sample`
  TensorShape<DynamicDimensions> operator[](int64_t sample) const {
    return tensor_shape<DynamicDimensions>(sample);
  }
  //gcc complains about constexpr
  /*constexpr*/ int sample_dim() const { return dim; }
  int size() const { return shapes.size() / sample_dim(); }

  int dim;
  using Base::shapes;

  /// @brief Convert to static TensorListShape
  ///
  /// Behaviour is undefined for other_ndim != sample_dim()
  /// @tparam other_ndim must be equal sample_dim()
  template <int other_ndim>
  TensorListShape<other_ndim> to_static() const & {
    static_assert(other_ndim != DynamicDimensions,
                  "Conversion to static only allowed for static shape");
    assert(sample_dim() == other_ndim && "Cannot convert to other ndim");
    return {shapes};
  }

  template <int other_ndim>
  TensorListShape<other_ndim> to_static() && {
    static_assert(other_ndim != DynamicDimensions,
                  "Conversion to static only allowed for static shape");
    assert(sample_dim() == other_ndim && "Cannot convert to other ndim");
    return {std::move(shapes)};
  }
};

template <int sample_ndim>
struct TensorListShape : TensorListShapeBase<TensorListShape<sample_ndim>, sample_ndim> {
  using Base = TensorListShapeBase<TensorListShape<sample_ndim>, sample_ndim>;
  TensorListShape() = default;
  TensorListShape(const TensorListShape &) = default;
  TensorListShape(TensorListShape &&) = default;
  TensorListShape(std::vector<TensorShape<sample_ndim>> sample_shapes)
      : Base(flatten_shapes(sample_shapes)) {}

  TensorListShape(const std::vector<int64_t> &shapes, int ndim = sample_ndim) : Base(shapes) {
    assert(ndim == sample_ndim);
  }
  TensorListShape(std::vector<int64_t> &&shapes, int ndim = sample_ndim) : Base(std::move(shapes)) {
    assert(ndim == sample_ndim);
  }

  /// @brief Return a static TensorShape for `sample`
  TensorShape<sample_ndim> operator[](int64_t sample) const {
    TensorShape<sample_ndim> result;
    int64_t base = sample_dim() * sample;
    for (int i = 0; i < sample_dim(); i++) {
      result[i] = shapes[base + i];
    }
    return result;
  }

  constexpr int sample_dim() const { return sample_ndim; }
  int size() const { return shapes.size() / sample_dim(); }

  using Base::shapes;

  template <int other_ndim>
  TensorListShape<other_ndim> to_static() const & {
    static_assert(other_ndim != sample_ndim, "Cannot convert to other static ndim");
    return {shapes};
  }

  template <int other_ndim>
  TensorListShape<other_ndim> to_static() && {
    static_assert(other_ndim != sample_ndim, "Cannot convert to other static ndim");
    return {std::move(shapes)};
  }
};

template <typename Derived, int sample_ndim>
template <int other_ndim>
TensorListShape<other_ndim> TensorListShapeBase<Derived, sample_ndim>::first() {
  static_assert(other_ndim <= sample_ndim || sample_ndim == DynamicDimensions,
                "Number of dimensions in subshape must be between 0 and sample_dim()");
  static_assert(other_ndim != DynamicDimensions, "This function can produce only static shapes");
  assert(0 <= other_ndim && other_ndim <= get_sample_dim() &&
         "Number of dimensions in subshape must be between 0 and sample_dim()");
  TensorListShape<other_ndim> result;
  result.shapes.resize(other_ndim * get_size());
  for (int sample = 0; sample < get_size(); sample++) {
    for (int d = 0; d < other_ndim; d++) {
      result.shapes[sample * other_ndim + d] = shapes[sample * get_sample_dim() + d];
    }
  }
  return result;
}

template <typename Derived, int sample_ndim>
template <int other_ndim>
TensorListShape<other_ndim> TensorListShapeBase<Derived, sample_ndim>::last() {
  static_assert(other_ndim <= sample_ndim || sample_ndim == DynamicDimensions,
                "Number of dimensions in subshape must be between 0 and sample_dim()");
  static_assert(other_ndim != DynamicDimensions, "This function can produce only static shapes");
  assert(0 <= other_ndim && other_ndim <= get_sample_dim() &&
         "Number of dimensions in subshape must be between 0 and sample_dim()");
  TensorListShape<other_ndim> result;
  result.shapes.resize(other_ndim * get_size());
  int start_offset = get_sample_dim() - other_ndim;
  for (int sample = 0; sample < get_size(); sample++) {
    for (int d = 0; d < other_ndim; d++) {
      result.shapes[sample * other_ndim + d] = shapes[sample * get_sample_dim() + start_offset + d];
    }
  }
  return result;
}

template <typename Derived, int sample_ndim>
TensorListShape<DynamicDimensions> TensorListShapeBase<Derived, sample_ndim>::first(int count) {
  assert(0 <= count && count <= get_sample_dim() &&
         "Number of dimensions in subshape must be between 0 and sample_dim()");
  TensorListShape<DynamicDimensions> result;
  result.shapes.resize(count * get_size());
  result.dim = count;
  for (int sample = 0; sample < get_size(); sample++) {
    for (int d = 0; d < count; d++) {
      result.shapes[sample * count + d] = shapes[sample * get_sample_dim() + d];
    }
  }
  return result;
}

template <typename Derived, int sample_ndim>
TensorListShape<DynamicDimensions> TensorListShapeBase<Derived, sample_ndim>::last(int count) {
  assert(0 <= count && count <= get_sample_dim() &&
         "Number of dimensions in subshape must be between 0 and sample_dim()");
  TensorListShape<DynamicDimensions> result;
  result.shapes.resize(count * get_size());
  result.dim = count;
  int start_offset = get_sample_dim() - count;
  for (int sample = 0; sample < get_size(); sample++) {
    for (int d = 0; d < count; d++) {
      result.shapes[sample * count + d] = shapes[sample * get_sample_dim() + start_offset + d];
    }
  }
  return result;
}

template <int left_ndim, int right_ndim>
bool operator==(const TensorListShape<left_ndim> &left, const TensorListShape<right_ndim> &right) {
  if (left.sample_dim() != right.sample_dim()) {
    return false;
  }
  return left.shapes == right.shapes;
}

template <int left_ndim, int right_ndim>
bool operator!=(const TensorListShape<left_ndim> &left, const TensorListShape<right_ndim> &right) {
  return !(left == right);
}

/// @brief Calculate offsets for Tensors stored in contigous buffer whose shapes
///        are described by tls. Offsets are calculated as number of elements of each tensors
///
/// @return std::vector<ptrdiff_t> containing tls.size() + 1 elements,
///         [i] is an offset to sample `i`, [i+1] is an offset one past sample `i`.
template <int sample_ndim>
std::vector<ptrdiff_t> calculate_offsets(const TensorListShape<sample_ndim> &tls) {
  std::vector<ptrdiff_t> offsets;
  offsets.resize(tls.size() + 1);
  offsets[0] = 0;
  for (int i = 0; i < tls.size(); i++) {
    auto sample_shape_span = tls.tensor_shape_span(i);
    offsets[i + 1] = offsets[i] + volume(sample_shape_span);
  }
  return offsets;
}

/// @brief Checks if all TensorShapes stored in `tls` have the same sizes
template <int ndim>
bool is_uniform(const TensorListShape<ndim> &tls) {
  if (!tls.size()) {
    return true;  // empty is uniform
  }
  auto first_span = tls.tensor_shape_span(0);
  for (int i = 1; i < tls.size(); i++) {
    if (first_span != tls.tensor_shape_span(i)) {
      return false;
    }
  }
  return true;
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SHAPE_H_
