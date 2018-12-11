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

namespace tensor {

// template <typename T>
// constexpr typename std::enable_if<std::is_fundamental<T>::value, size_t>::type ShapeDim(const T
// &) { return 1; }

// template <typename T, size_t N>
// constexpr int ShapeDim(T (&)[N]) { return int(N); }

// template <typename T>
// constexpr int ShapeDim(const T &t) {
//   return int(t.size());
// }

// /// @brief Calculates flat index of a given element in the tensor
// /// @remarks If pos has fewer dimensions than shape, the remaining offsets are assumed to be 0
// template <typename Shape, typename SamplePos>
// ptrdiff_t CalcOffset(const Shape &shape, const SamplePos &pos) {
//   ptrdiff_t ofs = pos[0];
//   const int m = ShapeDim(pos);
//   const int n = ShapeDim(shape);
//   int i;
//   for (i = 1; i < m; i++) {
//     ofs *= shape[i];
//     ofs += pos[i];
//   }
//   for (; i < n; i++) {
//     ofs *= shape[i];
//   }
//   return ofs;
// }

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

template <int ndim>
struct TensorShape;

/**
 * @brief Base class for TensorShape containing common code for iterators and operator[]
 */
template <typename Container>
struct TensorShapeBase {
  TensorShapeBase() : shape{} {}

  TensorShapeBase(const Container &c) : shape(c) {}
  TensorShapeBase(Container &&c) : shape(std::move(c)) {}
  using value_type = typename Container::value_type;
  using size_type = int;
  using reference = value_type &;
  using const_reference = const value_type &;
  using container_type = Container;
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

  size_type size() const { return shape.size(); }

  Container shape;
};

template <>
struct TensorShape<DynamicDimensions> : public TensorShapeBase<std::vector<int64_t>> {
  using Base = TensorShapeBase<std::vector<int64_t>>;
  TensorShape(const std::vector<int64_t> &s) : Base(s) {}

  template <size_t N>
  TensorShape(std::array<int64_t, N> s) : Base(typename Base::container_type(s.begin(), s.end())) {}

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

  // We allow only explicit conversion from dynamic to static dim
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

  TensorShape<DynamicDimensions> first(int count) {
    assert(0 <= count && count <= size() &&
           "Number of elements in subshape must be between 0 and size()");
    TensorShape<DynamicDimensions> result;
    result.shape.resize(count);
    for (int i = 0; i < count; i++) {
      result[i] = (*this)[i];
    }
    return result;
  }

  TensorShape<DynamicDimensions> last(int count) {
    assert(0 <= count && count <= size() &&
           "Number of elements in subshape must be between 0 and size()");
    TensorShape<DynamicDimensions> result;
    result.shape.resize(count);
    int start_offset = size() - count;
    for (int i = 0; i < count; i++) {
      result[i] = (*this)[start_offset + i];
    }
    return result;
  }
};

template <int ndim>
struct TensorShape : public TensorShapeBase<std::array<int64_t, ndim>> {
  using Base = TensorShapeBase<std::array<int64_t, ndim>>;
  TensorShape(const std::array<int64_t, ndim> &s) : Base(s) {}
  // We just want zero-initialization
  TensorShape() = default;
  // We allow only explicit operations on TensorShape static dim
  TensorShape(const TensorShape &) = default;
  TensorShape &operator=(const TensorShape &other) = default;

  template <typename... Ts>
  TensorShape(int64_t i0, Ts... s) : Base(typename Base::container_type{i0, int64_t{s}...}) {
    static_assert(sizeof...(Ts) == ndim - 1, "Number of shapes passed must match ndim");
  }

  template <int other_ndim>
  TensorShape<other_ndim> first() {
    static_assert(0 <= other_ndim && other_ndim <= ndim,
                  "Number of elements in subshape must be between 0 and size()");
    TensorShape<other_ndim> result;
    for (int i = 0; i < other_ndim; i++) {
      result[i] = (*this)[i];
    }
    return result;
  }

  template <int other_ndim>
  TensorShape<other_ndim> last() {
    static_assert(0 <= other_ndim && other_ndim <= ndim,
                  "Number of elements in subshape must be between 0 and size()");
    TensorShape<other_ndim> result;
    int start_offset = ndim - other_ndim;
    for (int i = 0; i < other_ndim; i++) {
      result[i] = (*this)[start_offset + i];
    }
    return result;
  }
};

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

template <int left_ndim, int right_ndim>
typename std::enable_if<left_ndim == DynamicDimensions || right_ndim == DynamicDimensions,
                        TensorShape<DynamicDimensions>>::type
shape_cat(const TensorShape<left_ndim> &left, const TensorShape<right_ndim> &right) {
  TensorShape<DynamicDimensions> result;
  int total_size = left.size() + right.size();
  result.shape.resize(total_size);
  for (int i = 0; i < left.size(); i++) {
    result[i] = left[i];
  }
  for (int i = 0; i < right.size(); i++) {
    result[left.size() + i] = right[i];
  }
  return result;
}

template <int left_ndim, int right_ndim>
typename std::enable_if<left_ndim != DynamicDimensions && right_ndim != DynamicDimensions,
                        TensorShape<left_ndim + right_ndim>>::type
shape_cat(const TensorShape<left_ndim> &left, const TensorShape<right_ndim> &right) {
  TensorShape<left_ndim + right_ndim> result;
  for (int i = 0; i < left_ndim; i++) {
    result[i] = left[i];
  }
  for (int i = 0; i < right_ndim; i++) {
    result[left_ndim + i] = right[i];
  }
  return result;
}

template <int sample_ndim>
typename std::enable_if<sample_ndim != DynamicDimensions, std::vector<int64_t>>::type
flatten_shapes(const std::vector<TensorShape<sample_ndim>> &shapes) {
  std::vector<int64_t> result;
  result.resize(sample_ndim * shapes.size());
  for (int sample = 0; sample < shapes.size(); sample++) {
    for (int axis = 0; axis < sample_ndim; axis++) {
      result[sample * sample_ndim + axis] = shapes[sample][axis];
    }
  }
  return result;
}

template <typename T>
typename std::enable_if<std::is_same<T, TensorShape<DynamicDimensions>>::value ||
                            std::is_same<T, std::vector<int64_t>>::value,
                        std::vector<int64_t>>::type
flatten_shapes(const std::vector<T> &shapes, int uniform_sample_ndim) {
  std::vector<int64_t> result;
  result.resize(uniform_sample_ndim * shapes.size());
  for (int sample = 0; sample < shapes.size(); sample++) {
    assert(shapes[sample].size() == uniform_sample_ndim);
    for (int axis = 0; axis < uniform_sample_ndim; axis++) {
      result[sample * uniform_sample_ndim + axis] = shapes[sample][axis];
    }
  }
  return result;
}

template <int sample_ndim>
struct TensorListShape;

template <>
struct TensorListShape<DynamicDimensions> {
  TensorListShape() : shapes(), dim(0) {}

  TensorListShape(const TensorListShape &) = default;
  TensorListShape(TensorListShape &&) = default;

  template <int other_sample_ndim>
  TensorListShape(const TensorListShape<other_sample_ndim> &other)
      : shapes(other.shapes), dim(other_sample_ndim) {}

  template <int other_sample_ndim>
  TensorListShape(TensorListShape<other_sample_ndim> &&other)
      : shapes(std::move(other.shapes)), dim(other_sample_ndim) {}

  TensorListShape(const std::vector<std::vector<int64_t>> &sample_shapes, int uniform_sample_ndim)
      : shapes(flatten_shapes(sample_shapes, uniform_sample_ndim)), dim(uniform_sample_ndim) {}

  TensorListShape(const std::vector<TensorShape<DynamicDimensions>> &sample_shapes,
                  int uniform_sample_ndim)
      : shapes(flatten_shapes(sample_shapes, uniform_sample_ndim)), dim(uniform_sample_ndim) {}

  // TODO(klecki): not sure if we should allow to create static tensor shapes direclty from dynamic
  // list shape
  template <int tensor_ndim>
  TensorShape<tensor_ndim> tensor_shape(int64_t sample) const {
    assert(tensor_ndim == sample_dim());
    TensorShape<tensor_ndim> out;
    if (tensor_ndim == DynamicDimensions) {
      out.shape.resize(sample_dim());
    }
    int64_t base = sample_dim() * sample;
    for (int i = 0; i < sample_dim(); i++) {
      out[i] = shapes[base + i];
    }
    return out;
  }

  span<int64_t> tensor_shape_span(int64_t sample) const {
    return {&shapes[sample * sample_dim()], static_cast<size_t>(sample_dim())};
  }

  TensorShape<DynamicDimensions> operator[](int64_t sample) const {
    return tensor_shape<DynamicDimensions>(sample);
  }

  constexpr int sample_dim() const { return dim; }

  int samples() const { return shapes.size() / sample_dim(); }

  std::vector<int64_t> shapes;
  int dim;

  // todo as static shape?
};

// TODO should we be able to only output a TensorShape of ndim=sample_dim for all cases of
// TensorListShape?
template <int sample_ndim>
struct TensorListShape {
  TensorListShape(std::vector<TensorShape<sample_ndim>> sample_shapes)
      : shapes(flatten_shapes(sample_shapes)) {}

  template <int tensor_ndim = sample_ndim>
  typename std::enable_if<tensor_ndim == sample_ndim || tensor_ndim == DynamicDimensions,
                          TensorShape<tensor_ndim>>::type
  tensor_shape(int64_t sample) const {
    TensorShape<tensor_ndim> out;
    if (tensor_ndim == DynamicDimensions) {
      out.shape.resize(sample_dim());
    }
    int64_t base = sample_dim() * sample;
    for (int i = 0; i < sample_dim(); i++) {
      out[i] = shapes[base + i];
    }
    return out;
  }

  span<int64_t, sample_ndim> tensor_shape_span(int64_t sample) const {
    return {&shapes[sample * sample_dim()]};
  }

  TensorShape<sample_ndim> operator[](int64_t sample) const {
    TensorShape<sample_ndim> result;
    int64_t base = sample_dim() * sample;
    for (int i = 0; i < sample_dim(); i++) {
      result[i] = shapes[base + i];
    }
    return result;
  }

  constexpr int sample_dim() const { return sample_ndim; }

  int samples() const { return shapes.size() / sample_dim(); }

  std::vector<int64_t> shapes;
};

template <int sample_ndim>
std::vector<ptrdiff_t> calculate_offsets(const TensorListShape<sample_ndim> &tls) {
  std::vector<ptrdiff_t> offsets;
  offsets.resize(tls.samples() + 1);
  offsets[0] = 0;
  for (int i = 0; i < tls.samples(); i++) {
    auto sample_shape_span = tls.tensor_shape_span(i);
    offsets[i] = volume(sample_shape_span);
  }
  return offsets;
}

// TODO:
// * `T* at(...)` as free function
// * discarding left and right elements of the ShapeList
// * zero filling?

}  // namespace tensor

#endif  // DALI_KERNELS_SHAPE_H_
