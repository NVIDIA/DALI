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

#ifndef DALI_KERNELS_TENSOR_VIEW_H_
#define DALI_KERNELS_TENSOR_VIEW_H_

#include "shape.h"

namespace tensor {

struct EmptyBackendTag {};

template <typename Backend, typename DataType, int ndim>
struct TensorView;

template <typename Backend, typename DataType>
struct TensorView<Backend, DataType, DynamicTensorShape> {
  TensorView() : data(nullptr), shape() {}

  template <int dim>
  TensorView(DataType *data, TensorShape<dim> shape) : data(data), shape(shape) {}
  TensorView(const TensorView &) = default;
  TensorView(TensorView &&) = default;
  TensorView &operator=(const TensorView &) = default;
  TensorView &operator=(TensorView &&) = default;

  // Dynamic accepts anything
  template <int other_ndim>
  TensorView(const TensorView<Backend, DataType, other_ndim> &other)
      : data(other.data), shape(other.shape) {}

  template <int other_ndim>
  TensorView(TensorView<Backend, DataType, other_ndim> &&other)
      : data(other.data), shape(other.shape) {}

  template <int other_ndim>
  TensorView &operator=(const TensorView<Backend, DataType, other_ndim> &other) {
    data = other.data;
    shape = other.shape;
    return *this;
  }

  template <int other_ndim>
  TensorView &operator=(TensorView<Backend, DataType, other_ndim> &&other) {
    data = std::move(other.data);  // TODO(klecki), should I null it?
    shape = std::move(other.shape);
    return *this;
  }

  // template <int other_ndim>
  // explicit operator TensorView<Backend, DataType, other_ndim>() {
  //   return {data, shape.to_static<other_ndim>()};
  // }

  template <int other_ndim>
  TensorView<Backend, DataType, other_ndim> to_static() {
    static_assert(other_ndim != DynamicTensorShape,
                  "Conversion to static only allowed for static shape");
    return {data, shape.to_static<other_ndim>()};
  }

  template <int other_ndim>
  TensorView<Backend, DataType, other_ndim> to_static(const TensorShape<other_ndim> &new_shape) {
    static_assert(other_ndim != DynamicTensorShape,
                  "Conversion to static only allowed for static shape");
    return {data, new_shape};
  }

  template <int other_ndim>
  TensorView<Backend, DataType, other_ndim> to_static(TensorShape<other_ndim> &&new_shape) {
    static_assert(other_ndim != DynamicTensorShape,
                  "Conversion to static only allowed for static shape");
    return {data, std::move(new_shape)};
  }

  DataType *data;
  TensorShape<DynamicTensorShape> shape;

  static const bool has_static_dim = false;
  int dim() const { return shape.size(); }

  // static const int static_dim = ndim;
  // int dim() const { return ShapeDim(shape); }

  // template <typename Offset>
  // DataType *at(const Offset &pos) const {
  //   return data + CalcOfffset(shape, pos);
  // }

  // template <typename... Indices>
  // DataType *at(int64_t idx0, Indices &&... idx) const {
  //   return data + CalcOfffset(shape, {idx0, (static_cast<int64_t>(idx))...});
  // }
};

template <typename Backend, typename DataType, int ndim>
struct TensorView {
  TensorView() : data(nullptr), shape() {}
  TensorView(DataType *data, TensorShape<ndim> shape) : data(data), shape(shape) {}
  TensorView(const TensorView &) = default;
  TensorView(TensorView &&) = default;
  TensorView &operator=(const TensorView &) = default;
  TensorView &operator=(TensorView &&) = default;

  DataType *data;
  TensorShape<ndim> shape;

  static const bool has_static_dim = false;
  constexpr int dim() const { return shape.size(); }

  // static const int static_dim = ndim;
  // int dim() const { return ShapeDim(shape); }

  // template <typename Offset>
  // DataType *at(const Offset &pos) const {
  //   return data + CalcOfffset(shape, pos);
  // }

  // template <typename... Indices>
  // DataType *at(int64_t idx0, Indices &&... idx) const {
  //   return data + CalcOfffset(shape, {idx0, (static_cast<int64_t>(idx))...});
  // }
};

template <typename Backend, typename DataType, int sample_ndim>
struct TensorListView;

template <typename Backend, typename DataType, int sample_ndim>
struct TensorListView {
  TensorListView() : data(nullptr), shape(), offsets() {}
  TensorListView(DataType *data, const std::vector<TensorShape<sample_ndim>> &shapes) : data(data), shape(shapes), offsets(calculate_offsets(shape)) {}

  DataType *data;
  TensorListShape<sample_ndim> shape;
  std::vector<ptrdiff_t> offsets;

  template <int dim = sample_ndim>
  TensorView<Backend, DataType, dim> operator[](int64_t sample) const {
    return {data + offsets[sample], shape.tensor_shape(sample)};
  }
  //TODO:
  // * size utilities
  // * dynamic dim handling
  // * generic `at` free function
};

// template <typename Backend, typename DataType, int sample_ndim>
// struct TensorListView : TensorListDim<sample_ndim> {
//   DataType *data = nullptr;
//   vector<int64_t> shape;
//   std::vector<ptrdiff_t> offsets;
//   using TensorListDim<sample_ndim>::sample_dim;
//   using TensorListDim<sample_ndim>::set_sample_dim;

//   size_t total_size() const { return offsets.empty() ? size_t(0) : offsets[num_samples()]; }

//   const int64_t num_samples() const { return shape.size() / sample_dim(); }

//   TensorListView(DataType *data, const vector<TensorShape<sample_ndim>> &shapes) : data(data) {
//     if (sample_ndim == -1 && !shapes.empty()) set_sample_dim(shapes[0]);
//     shapes.resize(sample_dim() * shapes.size());
//     for (size_t i = 0; i < shapes.size(); i++) {
//       assert(shapes[i].size() == sample_dim() &&
//              "All tensors in a tensor list must have same number of dimensions");
//     }
//     UpdateOffsets();
//   }

//   template <int dim = sample_ndim>
//   TensorShape<dim> tensor_shape(int64_t sample) const {
//     static_assert(sample_ndim < 0 || dim < 0 || sample_ndim == dim,
//                   "Mismatched number of dimensions");
//     if (sample_ndim < 0 && dim >= 0) {
//       assert(dim == sample_dim());
//     }

//     TensorShape<dim> out;
//     if (dim == -1) {
//       out.resize(sample_dim());
//     }
//     int64_t base = sample_dim() * sample;
//     for (int i = 0; i < sample_dim(); i++) {
//       out[i] = shape[base + i];
//     }
//     return out;
//   }

//   span<int64_t, sample_ndim> tensor_shape_span(int64_t sample) const {
//     return {&shape[sample * sample_dim()], sample_dim()};
//   }

//   void UpdateOffsets() {
//     offsets.resize(num_samples() + 1);
//     ptrdiff_t offset = 0;
//     auto d = sample_dim();
//     for (int i = 0; i < num_samples(); i++) {
//       offsets[i] = offset;
//       auto s = tensor_shape_span(i);
//       int64_t v = s[0];
//       for (int j = 1; j < s.size(); j++) v *= s[j];
//       offset += v;
//     }
//     offsets.back() = offset;
//   }

//   template <int dim = sample_ndim>
//   TensorView<Backend, DataType, dim> operator[](int64_t sample) const {
//     return {data + offsets[sample], tensor_shape(sample)};
//   }

//   // template <typename Offset>
//   // DataType *at(int64_t sample, const Offset &pos) const {
//   //   DALI_ENFORCE(ShapeDim(pos) < sample_dim());
//   //   return data + offsets[sample] + CalcOfffset(tensor_shape_span(sample), pos);
//   // }

//   // template <typename... Indices>
//   // DataType *at(int64_t sample, int64_t idx0, Indices &&... idx) const {
//   //   return data + offsets[sample] +
//   //          CalcOfffset(tensor_shape_span(sample), {idx0, (static_cast<int64_t>(idx))...});
//   // }
// };

}  // namespace tensor

#endif  // DALI_KERNELS_TENSOR_VIEW_H_
