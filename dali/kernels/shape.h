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

// /// @brief Returns the product of all elements in shapeZ
// /// @param shape - shape of a tensor whose elements we count
// template <typename Shape>
// inline int64_t Volume(const Shape &shape) {
//   int n = ShapeDim(shape);
//   if (n < 1)
//     return 0;
//   int64_t v = shape[0];
//   for (int i = 1; i < n; i++) {
//     v *= shape[i];
//   }
//   return v;
// }

constexpr int DynamicTensorShape = -1;

template <int ndim>
struct TensorShape;

template <>
struct TensorShape<DynamicTensorShape> {
  TensorShape(const std::vector<int64_t> &s) : shape(s) {}
  TensorShape(std::vector<int64_t> &&s) : shape(std::move(s)) {}
  template <size_t N>
  TensorShape(std::array<int64_t, N> s) : shape(s.begin(), s.end()) {}

  TensorShape() = default;
  TensorShape(const TensorShape &) = default;
  TensorShape(TensorShape &&) = default;
  TensorShape &operator=(const TensorShape &other) = default;
  TensorShape &operator=(TensorShape &&other) = default;

  // TODO(klecki) efficient size calculation for this vector?
  template <int other_ndim>
  TensorShape(const TensorShape<other_ndim> &other) : shape(other.shape.begin(), other.shape.end()) {
    static_assert(other_ndim != DynamicTensorShape, "This constructor should be not used");
  }

  template <int other_ndim>
  TensorShape(TensorShape<other_ndim> &&other) : shape(other.shape.begin(), other.shape.end()) {}

  template <int other_ndim>
  TensorShape &operator=(const TensorShape<other_ndim> &other) {
    shape = std::vector<int64_t>(other.shape.begin(), other.shape.end());
    return *this;
  }

  template <int other_ndim>
  TensorShape &operator=(TensorShape<other_ndim> &&other) {
    shape = std::vector<int64_t>(other.shape.begin(), other.shape.end());
    return *this;
  }

  // template <int other_ndim>
  // TensorShape(const TensorShape<other_ndim> &other) {
  //   *this = other;
  // }

  // template <int other_ndim>
  // TensorShape &operator=(const TensorShape<other_ndim> &other) {
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
  // template <int other_ndim>
  // explicit operator TensorShape<other_ndim>() const;
  template <int other_ndim>
  TensorShape<other_ndim> to_static() const;

  int64_t &operator[](int d) { return shape[d]; }
  const int64_t &operator[](int d) const { return shape[d]; }

  std::vector<int64_t> shape;
  int size() const { return shape.size(); }
};

template <int ndim>
struct TensorShape {
  TensorShape(const std::array<int64_t, ndim> &s) : shape(s) {
    std::cout << "const std::array<int64_t, ndim> &s" << std::endl;
  }
  TensorShape(std::array<int64_t, ndim> &&s) : shape(std::move(s)) {
    std::cout << "std::array<int64_t, ndim> &&s" << std::endl;
  }

  // We just want zero-initialization
  TensorShape() : shape{} {}
  // We allow only explicit operations on TensorShape static dim
  TensorShape(const TensorShape &) = default;
  TensorShape(TensorShape &&) = default;
  TensorShape &operator=(const TensorShape &other) = default;
  TensorShape &operator=(TensorShape &&other) = default;

  template <typename... Ts>
  TensorShape(int64_t i0, Ts... s) : shape{i0, int64_t{s}...} {
    static_assert(sizeof...(Ts) == ndim - 1, "Number of shapes passed must match ndim");
    std::cout << "TensorShape(int64_t i0, Ts... s)" << std::endl;
    // TODO static_assert for Ts == int64_t
  }

  int64_t &operator[](int d) { return shape[d]; }
  const int64_t &operator[](int d) const { return shape[d]; }

  std::array<int64_t, ndim> shape;
  constexpr int size() const { return ndim; }
};

// template<int other_ndim>
// TensorShape<DynamicTensorShape>::operator TensorShape<other_ndim>() const {
//   std::cout << "operator TensorShape<other_ndim>()" << std::endl;
//   assert(size() == other_ndim);
//   TensorShape<other_ndim> shape;
//   for (int i = 0; i < other_ndim; i++) {
//     shape[i] = (*this)[i];
//   }
//   return shape;
// }

template <int other_ndim>
TensorShape<other_ndim> TensorShape<DynamicTensorShape>::to_static() const {
  std::cout << "operator TensorShape<other_ndim>()" << std::endl;
  assert(size() == other_ndim);
  TensorShape<other_ndim> shape;
  for (int i = 0; i < other_ndim; i++) {
    shape[i] = (*this)[i];
  }
  return shape;
}

// TODO should we be able to only output a tensor of sample_dim dim for all cases of TensorListShape
template <int sample_ndim>
struct TensorListShape {
  TensorListShape(std::vector<TensorShape<sample_ndim>> sample_shapes)
      : shapes(std::accumulate(sample_shapes.begin(), sample_shapes.end(), std::vector<int64_t>(),
                               [](std::vector<int64_t> current, const TensorShape<sample_ndim> &next) {
                                 return std::move(current).insert(current.end(), next.shape.begin(),
                                                                  next.shape.end());
                               })) {}

  template <int tensor_dim = sample_ndim>
  typename std::enable_if<tensor_dim == sample_ndim || tensor_dim == DynamicTensorShape,
                          TensorShape<tensor_dim>>::type
  tensor_shape(int64_t sample) const {
    TensorShape<tensor_dim> out;
    if (tensor_dim == DynamicTensorShape) {
      out.shape.resize(sample_dim());
    }
    int64_t base = sample_dim() * sample;
    for (int i = 0; i < sample_dim(); i++) {
      out[i] = shapes[base + i];
    }
    return out;
  }

  span<int64_t, sample_ndim> tensor_shape_span(int64_t sample) const {
    return {&shapes[sample * sample_dim()], sample_dim()};
  }

  constexpr int sample_dim() const { return sample_ndim; }

  int samples() const { return shapes.size() / sample_dim(); }

  std::vector<int64_t> shapes;
};

template <>
struct TensorListShape<DynamicTensorShape> {
  template <int tensor_dim>
  TensorShape<tensor_dim> tensor_shape(int64_t sample) const {
    TensorShape<tensor_dim> out;
    if (tensor_dim == DynamicTensorShape) {
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

  int sample_dim() const { return dim; }

  int samples() const { return shapes.size() / sample_dim(); }

  std::vector<int64_t> shapes;
  int dim;
};

template <int sample_dim>
std::vector<ptrdiff_t> calculate_offsets(const TensorListShape<sample_dim> &tls) {
  std::vector<ptrdiff_t> offsets;
  offsets.resize(tls.samples() + 1);
  offsets.push_back(0);
  for (int i = 0; i < tls.samples(); i++) {
    auto sample_shape_span = tls.tensor_shape_span(i);
    auto off = std::accumulate(sample_shape_span.begin(), sample_shape_span.end(),
                               static_cast<ptrdiff_t>(1),
                               [](const ptrdiff_t &a, const ptrdiff_t &b) { return a * b; });
    offsets.push_back(off);
  }
  return offsets;
}

// TODO:
// * `multiply(span)`
// * `flatten` to vector
// * `T* at(...)` as free function


// template <int sample_ndim>
// struct TensorListDim {
//   inline constexpr int sample_dim() const { return sample_ndim; }

//  protected:
//   inline void set_sample_dim(int dim) { assert(dim == sample_ndim); }
// };

// template <>
// struct TensorListDim<-1> {
//   inline constexpr int sample_dim() const { return sample_ndim; }

//  protected:
//   inline void set_sample_dim(int dim) { sample_ndim = dim; }
//   int sample_ndim;
// };

}  // namespace tensor

#endif  // DALI_KERNELS_SHAPE_H_
