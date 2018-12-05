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

#include <dali/common.h>
#include <vector>
#include <array>
#include <cassert>

namespace dali {

template <typename T>
constexpr typename std::enable_if<std::is_fundamental<T>::value, size_t>::type ShapeDim(const T &)
{ return 1; }

template <typename T, size_t N>
constexpr int ShapeDim(T (&)[N]) { return int(N); }

template <typename T, size_t N>
constexpr int ShapeDim(const std::array<T, N> &) { return N; }

template <typename T, typename A>
constexpr int ShapeDim(const std::vector<T, A> &shape) { return int(shape.size()); }

template <typename T>
constexpr int ShapeDim(const std::initializer_list<T> &shape) { return shape.size(); }

/// @brief Calculates flat index of a given element in the tensor
/// @remarks If pos has fewer dimensions than shape, the remaining offsets are assumed to be 0
template <typename Shape, typename SamplePos>
ptrdiff_t CalcOffset(const Shape &shape, const SamplePos &pos) {
  ptrdiff_t ofs = pos[0];
  const int m = ShapeDim(pos);
  const int n = ShapeDim(shape);
  int i;
  for (i = 1; i < m; i++) {
    ofs *= shape[i];
    ofs += pos[i];
  }
  for (; i < n; i++) {
    ofs *= shape[i];
  }
  return ofs;
}

/// @brief Returns the product of all elements in shapeZ
/// @param shape - shape of a tensor whose elements we count
template <typename Shape>
inline Index Volume(const Shape &shape) {
  int n = ShapeDim(shape);
  if (n < 1)
    return 0;
  Index v = shape[0];
  for (int i = 1; i < n; i++) {
    v *= shape[i];
  }
  return v;
}

template <int dim_>
struct TensorShape {
  TensorShape(std::array<Index, dim_> &&s = {}) : data(std::move(s)) {}

  Index &operator[](int d) { return data[d]; }
  Index operator[](int d) const { return data[d]; }
  std::array<Index, dim_> data;

  constexpr int size() const { return dim_; }

  template <int other_dim>
  TensorShape &operator=(const TensorShape<other_dim> &other) {
    assert(other.size() == dim_ && "Cannot assigned a tensor of different dimensionality to a fixed-size tensor");
    for (int i = 0; i < dim_; i++) {
      data[i] = other[i];
    }
    return *this;
  }
};

template <>
struct TensorShape<-1> {
  TensorShape() = default;
  template <int other_dim>
  TensorShape(const TensorShape<other_dim> &other) {
    *this = other;
  }
  TensorShape(std::vector<Index> s) : data(std::move(s)) {}

  Index &operator[](int d) { return data[d]; }
  Index operator[](int d) const { return data[d]; }
  std::vector<Index> data;  // TODO: replace with something mo

  const int size() const { return data.size(); }

  template <int other_dim>
  TensorShape &operator=(const TensorShape<other_dim> &other) {
    data.resize(other.size());
    for (int i = 0; i < size(); i++) {
      data[i] = other[i];
    }
    return *this;
  }

  TensorShape &operator==(TensorShape<-1> &&other) {
    data = std::move(other.data);
    return *this;
  }
};

template <int sample_dim_>
struct TensorListDim {
  inline constexpr int sample_dim() const { return sample_dim_; }
protected:
  inline void set_sample_dim(int dim) {
    assert(dim == sample_dim_);
  }
};

template <>
struct TensorListDim<-1> {
  inline constexpr int sample_dim() const { return sample_dim_; }
protected:
  inline void set_sample_dim(int dim) {
    sample_dim_ = dim;
  }
  int sample_dim_;
};

template <typename T, size_t static_size = (size_t)-1>
struct span {
  span() = default;
  span(T *p, size_t n = static_size) : p(p) { assert(n == static_size); }

  const T *p = nullptr;

  constexpr T *base() const { return p; }
  constexpr T &operator[](ptrdiff_t index) const { return p[index]; }
  constexpr T *begin() const { return p; }
  constexpr T *end() const { return p + size(); }
  constexpr size_t size() const { return static_size; }
};

template <typename T>
struct span<T, (size_t)-1> {
  span() = default;
  span(T *p, size_t n) : p(p), n(n) {}

  const T *p = nullptr;
  const size_t n = 0;

  constexpr T *base() const { return p; }
  constexpr T &operator[](ptrdiff_t index) const { return p[index]; }
  constexpr T *begin() const { return p; }
  constexpr T *end() const { return p + size(); }
  constexpr size_t size() const { return n; }
};


}  // namespace dali

#endif   // DALI_KERNELS_SHAPE_H_
