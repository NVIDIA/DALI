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
#include <array>
#include <cassert>
#include <vector>
#include <iostream>
namespace dali {

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

// /// @brief Returns the product of all elements in shapeZ
// /// @param shape - shape of a tensor whose elements we count
// template <typename Shape>
// inline Index Volume(const Shape &shape) {
//   int n = ShapeDim(shape);
//   if (n < 1)
//     return 0;
//   Index v = shape[0];
//   for (int i = 1; i < n; i++) {
//     v *= shape[i];
//   }
//   return v;
// }

constexpr int DynamicTensorShape = -1;

template <int dim_>
struct TensorShape;

template <>
struct TensorShape<DynamicTensorShape> {
  TensorShape(const std::vector<Index> &s) : shape(s) {}
  TensorShape(std::vector<Index> &&s) : shape(std::move(s)) {}
  template <size_t N>
  TensorShape(std::array<int64_t, N> s) : shape(s.begin(), s.end()) {}

  TensorShape() = default;
  TensorShape(const TensorShape &) = default;
  TensorShape(TensorShape &&) = default;
  TensorShape& operator=(const TensorShape &other) = default;
  TensorShape& operator=(TensorShape &&other) = default;

  // TODO(klecki) efficient size calculation for this vector?
  template <int other_dim>
  TensorShape(const TensorShape<other_dim> &other) : shape(other.shape.begin(), other.shape.end()) {}

  template <int other_dim>
  TensorShape(TensorShape<other_dim> &&other) : shape(other.shape.begin(), other.shape.end()) {}

  template <int other_dim>
  TensorShape& operator=(const TensorShape<other_dim> &other) {
    shape = std::vector<int64_t>(other.shape.begin(), other.shape.end());
    return *this;
  }

  template <int other_dim>
  TensorShape& operator=(TensorShape<other_dim> &&other) {
    shape = std::vector<int64_t>(other.shape.begin(), other.shape.end());
    return *this;
  }

  // template <int other_dim>
  // TensorShape(const TensorShape<other_dim> &other) {
  //   *this = other;
  // }


  // template <int other_dim>
  // TensorShape &operator=(const TensorShape<other_dim> &other) {
  //   data.resize(other.size());
  //   for (int i = 0; i < size(); i++) {
  //     data[i] = other[i];
  //   }
  //   return *this;
  // }

  // TensorShape &operator=(TensorShape<-1> &&other) {
  //   data = std::move(other.data);
  //   return *this;
  // }

  // We allow only explicit conversion from dynamic to static dim
  template <int other_dim>
  explicit operator TensorShape<other_dim>() const;

  int64_t &operator[](int d) { return shape[d]; }
  const int64_t &operator[](int d) const { return shape[d]; }

  std::vector<int64_t> shape;
  const int size() const { return shape.size(); }
};

template <int dim_>
struct TensorShape {
  TensorShape(const std::array<int64_t, dim_> &s) : shape(s) {std::cout << "const std::array<int64_t, dim_> &s" << std::endl;}
  TensorShape(std::array<int64_t, dim_> &&s) : shape(std::move(s)) {std::cout << "std::array<int64_t, dim_> &&s" << std::endl;}

  // We just want zero-initialization
  TensorShape() : shape{} {}
  // We allow only explicit operations on TensorShape static dim
  TensorShape(const TensorShape &) = default;
  TensorShape(TensorShape &&) = default;
  TensorShape& operator=(const TensorShape &other) = default;
  TensorShape& operator=(TensorShape &&other) = default;


  template <typename... Ts>
  TensorShape(int64_t i0, Ts... s) : shape{i0, int64_t{s}...} {
    static_assert(sizeof...(Ts) == dim_ - 1, "Number of shapes passed must match dim_");
    std::cout << "TensorShape(int64_t i0, Ts... s)" << std::endl;
    //TODO static_assert for Ts == int64_t
  }

  int64_t &operator[](int d) { return shape[d]; }
  const int64_t &operator[](int d) const { return shape[d]; }

  std::array<int64_t , dim_> shape;
  constexpr int size() const { return dim_; }
};

template<int other_dim>
TensorShape<DynamicTensorShape>::operator TensorShape<other_dim>() const {
  std::cout << "operator TensorShape<other_dim>()" << std::endl;
  assert(size() == other_dim);
  TensorShape<other_dim> shape;
  for (int i = 0; i < other_dim; i++) {
    shape[i] = (*this)[i];
  }
  return shape;
}

template <int sample_dim_>
struct TensorListDim {
  inline constexpr int sample_dim() const { return sample_dim_; }

 protected:
  inline void set_sample_dim(int dim) { assert(dim == sample_dim_); }
};

template <>
struct TensorListDim<-1> {
  inline constexpr int sample_dim() const { return sample_dim_; }

 protected:
  inline void set_sample_dim(int dim) { sample_dim_ = dim; }
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

#endif  // DALI_KERNELS_SHAPE_H_
