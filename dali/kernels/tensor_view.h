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

template <typename T>
constexpr typename std::enable_if<std::is_fundamental<T>::value, size_t>::type ShapeDim(const T &) {
  return 1;
}

template <typename T, size_t N>
constexpr int ShapeDim(T (&)[N]) {
  return int(N);
}

template <typename T>
constexpr int ShapeDim(const T &t) {
  return int(t.size());
}

template <typename Shape, typename Position>
bool ContainsCoords(const Shape &shape, const Position &pos) {
  const int shape_dim = ShapeDim(shape);
  const int pos_dim = ShapeDim(pos);
  if (pos_dim > shape_dim) {
    return false;
  }
  for (int i = 0; i < pos_dim; i++) {
    if (pos[i] > shape[i]) {
      return false;
    }
  }
  return true;
}

/// @brief Calculates flat index of a given element in the tensor
/// @remarks If pos has fewer dimensions than shape, the remaining offsets are assumed to be 0
template <typename Shape, typename Position>
ptrdiff_t CalcOffset(const Shape &shape, const Position &pos) {
  ptrdiff_t ofs = pos[0];
  const int pos_dim = ShapeDim(pos);
  const int shape_dim = ShapeDim(shape);
  int i;
  for (i = 1; i < pos_dim; i++) {
    ofs *= shape[i];
    ofs += pos[i];
  }
  for (; i < shape_dim; i++) {
    ofs *= shape[i];
  }
  return ofs;
}

struct EmptyBackendTag {};

template <typename Backend, typename DataType, int ndim>
struct TensorView;

template <typename Backend, typename DataType, int ndim>
struct TensorViewBase {
  int dim() const { return shape.size(); }

  template <typename... Indices>
  DataType *operator()(int64_t idx0, Indices &&... idx) const {
    return data + CalcOfffset(shape, {idx0, (int64_t{idx})...});
  }

  template <typename Offset>
  DataType *operator()(const Offset &pos) const {
    return data + CalcOfffset(shape, pos);
  }

  template <int other_ndim>
  TensorView<Backend, DataType, other_ndim> to_static();

  template <int other_ndim>
  TensorView<Backend, DataType, other_ndim> to_static(const TensorShape<other_ndim> &new_shape);

  template <int other_ndim>
  TensorView<Backend, DataType, other_ndim> to_static(TensorShape<other_ndim> &&new_shape);

  DataType *data = nullptr;
  TensorShape<ndim> shape;

 protected:
  TensorViewBase() = default;
  TensorViewBase(const TensorViewBase &) = default;
  TensorViewBase(TensorViewBase &&) = default;
  TensorViewBase(DataType *data, const TensorShape<ndim> &shape) : data(data), shape(shape) {}
  TensorViewBase(DataType *data, TensorShape<ndim> &&shape) : data(data), shape(std::move(shape)) {}
};

template <typename Backend, typename DataType>
struct TensorView<Backend, DataType, DynamicDimensions>
    : TensorViewBase<Backend, DataType, DynamicDimensions> {
  using Base = TensorViewBase<Backend, DataType, DynamicDimensions>;

  TensorView() = default;

  template <int ndim>
  TensorView(DataType *data, const TensorShape<ndim> &shape) : Base(data, shape) {}
  template <int ndim>
  TensorView(DataType *data, TensorShape<ndim> &&shape) : Base(data, std::move(shape)) {}
  TensorView(const TensorView &) = default;
  TensorView &operator=(const TensorView &) = default;
  TensorView(TensorView &&other) : Base(other.data, std::move(other.shape)) {
    other.data = nullptr;
  }
  TensorView &operator=(TensorView &&other) {
    this->data = other.data;
    other.data = nullptr;
    this->shape = std::move(other.shape);
    return *this;
  }

  // Dynamic accepts anything
  template <int other_ndim>
  TensorView(const TensorView<Backend, DataType, other_ndim> &other)
      : Base(other.data, other.shape) {}
  template <int other_ndim>
  TensorView(TensorView<Backend, DataType, other_ndim> &&other)
      : Base(other.data, std::move(other.shape)) {
    other.data = nullptr;
  }

  template <int other_ndim>
  TensorView &operator=(const TensorView<Backend, DataType, other_ndim> &other) {
    this->data = other.data;
    this->shape = other.shape;
    return *this;
  }

  template <int other_ndim>
  TensorView &operator=(TensorView<Backend, DataType, other_ndim> &&other) {
    this->data = other.data;
    other.data = nullptr;
    this->shape = std::move(other.shape);
    return *this;
  }
};

template <typename Backend, typename DataType, int ndim>
struct TensorView : TensorViewBase<Backend, DataType, ndim> {
  using Base = TensorViewBase<Backend, DataType, ndim>;
  TensorView() = default;
  TensorView(DataType *data, const TensorShape<ndim> &shape) : Base(data, shape) {}
  TensorView(DataType *data, TensorShape<ndim> &&shape) : Base(data, std::move(shape)) {}
  TensorView(const TensorView &) = default;
  TensorView &operator=(const TensorView &) = default;
};

template <typename Backend, typename DataType, int ndim>
template <int other_ndim>
TensorView<Backend, DataType, other_ndim> TensorViewBase<Backend, DataType, ndim>::to_static() {
  static_assert(other_ndim != DynamicDimensions,
                "Conversion to static only allowed for static shape");
  // assert(other_ndim == dim() && "Cannot convert to other ndim");
  return {data, shape.template to_static<other_ndim>()};
}

template <typename Backend, typename DataType, int ndim>
template <int other_ndim>
TensorView<Backend, DataType, other_ndim> TensorViewBase<Backend, DataType, ndim>::to_static(
    const TensorShape<other_ndim> &new_shape) {
  static_assert(other_ndim != DynamicDimensions,
                "Conversion to static only allowed for static shape");
  return {data, new_shape};
}

template <typename Backend, typename DataType, int ndim>
template <int other_ndim>
TensorView<Backend, DataType, other_ndim> TensorViewBase<Backend, DataType, ndim>::to_static(
    TensorShape<other_ndim> &&new_shape) {
  static_assert(other_ndim != DynamicDimensions,
                "Conversion to static only allowed for static shape");
  return {data, std::move(new_shape)};
}

template <typename Backend, typename DataType, int sample_ndim>
struct TensorListView;


template <typename Backend, typename DataType>
struct TensorListView<Backend, DataType, DynamicDimensions> {
  TensorListView() : data(nullptr), shape(), offsets() {}
  TensorListView(DataType *data, const std::vector<TensorShape<DynamicDimensions>> &shapes)
      : data(data), shape(shapes), offsets(calculate_offsets(shape)) {}

  TensorView<Backend, DataType, DynamicDimensions> operator[](int sample) const {
    return {data + offsets[sample], shape[sample]};
  }

  template <int other_sample_ndim>
  TensorListView<Backend, DataType, other_sample_ndim> as_static_ndim() {
    return {data, shape, offsets};
  }

  DataType *data;
  TensorListShape<DynamicDimensions> shape;
  std::vector<ptrdiff_t> offsets;
};

template <typename Backend, typename DataType, int sample_ndim>
struct TensorListView {
  TensorListView() : data(nullptr), shape(), offsets() {}
  TensorListView(DataType *data, const std::vector<TensorShape<sample_ndim>> &shapes)
      : data(data), shape(shapes), offsets(calculate_offsets(shape)) {}

  TensorListView(DataType *data, const TensorListShape<sample_ndim> &shape,
                 const std::vector<ptrdiff_t> &offsets)
      : data(data), shape(shape), offsets(offsets) {}

  TensorListView(DataType *data, TensorListShape<sample_ndim> &&shape,
                 std::vector<ptrdiff_t> &&offsets)
      : data(data), shape(std::move(shape)), offsets(std::move(offsets)) {}

  TensorView<Backend, DataType, sample_ndim> operator[](int sample) const {
    return {data + offsets[sample], shape[sample]};
  }

  DataType *data;
  TensorListShape<sample_ndim> shape;
  std::vector<ptrdiff_t> offsets;
};

// TODO:
// * size utilities
// * dynamic dim handling
// * generic `at` free function
// * range loop? we're creating TensorView on the fly

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
