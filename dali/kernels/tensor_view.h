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

#ifndef DALI_KERNELS_TENSOR_VIEW_
#define DALI_KERNELS_TENSOR_VIEW_

#include "shape.h"

namespace dali {

struct EmptyBackendTag {};

template <typename Backend, typename DataType, int dim_>
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
  template <int other_dim>
  TensorView(const TensorView<Backend, DataType, other_dim> &other) : data(other.data), shape(other.shape) {}

  template <int other_dim>
  TensorView(TensorView<Backend, DataType, other_dim> &&other) : data(other.data), shape(other.shape) {}

  template <int other_dim>
  TensorView& operator=(const TensorView<Backend, DataType, other_dim> &other) {
    data = other.data;
    shape = other.shape;
    return *this;
  }

  template <int other_dim>
  TensorView& operator=(TensorView<Backend, DataType, other_dim> &&other) {
    data = std::move(other.data); // TODO(klecki), should I null it?
    shape = std::move(other.shape);
    return *this;
  }

  template <int other_dim>
  explicit operator TensorView<Backend, DataType, other_dim>() {
    return {data, TensorShape<other_dim>(shape)};
  }

  DataType *data;
  TensorShape<DynamicTensorShape> shape;

  // static const int static_dim = dim_;
  // int dim() const { return ShapeDim(shape); }

  // template <typename Offset>
  // DataType *at(const Offset &pos) const {
  //   return data + CalcOfffset(shape, pos);
  // }

  // template <typename... Indices>
  // DataType *at(Index idx0, Indices &&... idx) const {
  //   return data + CalcOfffset(shape, {idx0, (static_cast<Index>(idx))...});
  // }
};

template <typename Backend, typename DataType, int dim_>
struct TensorView {
  TensorView() : data(nullptr), shape() {}
  TensorView(DataType *data, TensorShape<dim_> shape) : data(data), shape(shape) {}
  TensorView(const TensorView &) = default;
  TensorView(TensorView &&) = default;
  TensorView &operator=(const TensorView &) = default;
  TensorView &operator=(TensorView &&) = default;

  // template <int other_dim, typename = typename std::enable_if<other_dim == dim_ || dim_ == DynamicTensorShape >::type>
  // TensorView(const TensorView<Backend, DataType, other_dim> &other) : data(other.data), shape(other.shape) {}

  DataType *data;
  TensorShape<dim_> shape;

  // static const int static_dim = dim_;
  // int dim() const { return ShapeDim(shape); }

  // template <typename Offset>
  // DataType *at(const Offset &pos) const {
  //   return data + CalcOfffset(shape, pos);
  // }

  // template <typename... Indices>
  // DataType *at(Index idx0, Indices &&... idx) const {
  //   return data + CalcOfffset(shape, {idx0, (static_cast<Index>(idx))...});
  // }
};


// template <typename Backend, typename DataType, int dim_>
// template <int other_dim>
// TensorView<Backend, DataType, DynamicTensorShape>::TensorView(
//     const TensorView<Backend, DataType, other_dim> &other)
//     : data(other.data), shape(other.shape) {}

template <typename Backend, typename DataType, int sample_dim_>
struct TensorListView : TensorListDim<sample_dim_> {
  DataType *data = nullptr;
  vector<Index> shape;
  std::vector<ptrdiff_t> offsets;
  using TensorListDim<sample_dim_>::sample_dim;
  using TensorListDim<sample_dim_>::set_sample_dim;

  size_t total_size() const { return offsets.empty() ? size_t(0) : offsets[num_samples()]; }

  const Index num_samples() const { return shape.size() / sample_dim(); }

  TensorListView(DataType *data, const vector<TensorShape<sample_dim_>> &shapes) : data(data) {
    if (sample_dim_ == -1 && !shapes.empty()) set_sample_dim(shapes[0]);
    shapes.resize(sample_dim() * shapes.size());
    for (size_t i = 0; i < shapes.size(); i++) {
      assert(shapes[i].size() == sample_dim() &&
             "All tensors in a tensor list must have same number of dimensions");
    }
    UpdateOffsets();
  }

  template <int dim = sample_dim_>
  TensorShape<dim> tensor_shape(Index sample) const {
    static_assert(sample_dim_ < 0 || dim < 0 || sample_dim_ == dim,
                  "Mismatched number of dimensions");
    if (sample_dim_ < 0 && dim >= 0) {
      assert(dim == sample_dim());
    }

    TensorShape<dim> out;
    if (dim == -1) {
      out.resize(sample_dim());
    }
    Index base = sample_dim() * sample;
    for (int i = 0; i < sample_dim(); i++) {
      out[i] = shape[base + i];
    }
    return out;
  }

  span<Index, sample_dim_> tensor_shape_span(Index sample) const {
    return {&shape[sample * sample_dim()], sample_dim()};
  }

  void UpdateOffsets() {
    offsets.resize(num_samples() + 1);
    ptrdiff_t offset = 0;
    auto d = sample_dim();
    for (int i = 0; i < num_samples(); i++) {
      offsets[i] = offset;
      auto s = tensor_shape_span(i);
      Index v = s[0];
      for (int j = 1; j < s.size(); j++) v *= s[j];
      offset += v;
    }
    offsets.back() = offset;
  }

  template <int dim = sample_dim_>
  TensorView<Backend, DataType, dim> operator[](Index sample) const {
    return {data + offsets[sample], TensorShape<dim>(sample)};
  }

  template <typename Offset>
  DataType *at(Index sample, const Offset &pos) const {
    DALI_ENFORCE(ShapeDim(pos) < sample_dim());
    return data + offsets[sample] + CalcOfffset(tensor_shape_span(sample), pos);
  }

  template <typename... Indices>
  DataType *at(Index sample, Index idx0, Indices &&... idx) const {
    return data + offsets[sample] +
           CalcOfffset(tensor_shape_span(sample), {idx0, (static_cast<Index>(idx))...});
  }
};

}  // namespace dali

#endif  // DALI_KERNELS_TENSOR_VIEW_
