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

#include <type_traits>
#include <utility>
#include <vector>
#include "dali/kernels/tensor_shape.h"

namespace dali {
namespace kernels {

namespace detail {

template <typename From, typename To>
struct check_implicit_conversion {
  static_assert(std::is_convertible<From*, To*>::value, "Conversion impossible");
  static_assert(std::is_same<
    typename std::remove_cv<From>::type,
    typename std::remove_cv<To>::type>::value,
    "Implicit conversion can only change CV qualifiers");
};

}  // namespace detail

template <typename T>
constexpr typename std::enable_if<std::is_fundamental<T>::value, size_t>::type ShapeDim(const T &) {
  return 1;
}

template <typename T, size_t N>
constexpr int ShapeDim(T (&)[N]) {
  return static_cast<int>(N);
}

template <typename T>
constexpr int ShapeDim(const T &t) {
  return static_cast<int>(t.size());
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

/// @brief Non-owning wrapper for Tensor, containing typed pointer to data and TensorShape
///
/// @tparam Backend
/// @tparam DataType
/// @tparam ndim either static for ndim >= 0 or DynamicDimensions
template <typename Backend, typename DataType, int ndim = DynamicDimensions>
struct TensorView;

template <typename Backend, typename DataType, int ndim>
struct TensorViewBase {
  using element_type = DataType;
  int dim() const { return shape.size(); }

  /// @brief Utility to calculate pointer to element at given coordinates
  template <typename... Indices>
  DataType *operator()(int64_t idx0, Indices &&... idx) const {
    return data + CalcOffset(shape, std::array<ptrdiff_t, sizeof...(Indices) + 1>{
                                        idx0, (ptrdiff_t{idx})...});
  }

  /// @brief Utility to calculate pointer to element at given coordinates
  template <typename Offset>
  DataType *operator()(const Offset &pos) const {
    return data + CalcOffset(shape, pos);
  }

  template <int other_ndim>
  TensorView<Backend, DataType, other_ndim> to_static();

  DataType *data = nullptr;
  TensorShape<ndim> shape;

  ptrdiff_t num_elements() const {
    return volume(shape);
  }

 protected:
  TensorViewBase() = default;
  TensorViewBase(const TensorViewBase &) = default;
  TensorViewBase(DataType *data, const TensorShape<ndim> &shape) : data(data), shape(shape) {}
  TensorViewBase(DataType *data, TensorShape<ndim> &&shape) : data(data), shape(std::move(shape)) {}
};

/// @brief Dynamic TensorView can be constructed from any Static TensorView
template <typename Backend, typename DataType>
struct TensorView<Backend, DataType, DynamicDimensions>
    : TensorViewBase<Backend, DataType, DynamicDimensions> {
  using Base = TensorViewBase<Backend, DataType, DynamicDimensions>;

  TensorView() = default;

  TensorView(DataType *data, const TensorShape<DynamicDimensions> &shape)
      : Base(data, shape) {}

  TensorView(DataType *data, TensorShape<DynamicDimensions> &&shape)
      : Base(data, std::move(shape)) {}

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
  template <int other_ndim, typename U>
  TensorView(const TensorView<Backend, U, other_ndim> &other)
      : Base(other.data, other.shape) {
    detail::check_implicit_conversion<U, DataType>();
  }
  template <int other_ndim, typename U>
  TensorView(TensorView<Backend, U, other_ndim> &&other)
      : Base(other.data, std::move(other.shape)) {
    detail::check_implicit_conversion<U, DataType>();
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

  // Pointer promotion
  template <typename U>
  TensorView(const TensorView<Backend, U, ndim> &other)
      : Base(other.data, other.shape) {
    detail::check_implicit_conversion<U, DataType>();
  }
  // Pointer promotion
  template <typename U>
  TensorView(TensorView<Backend, U, ndim> &&other)
      : Base(other.data, std::move(other.shape)) {
    detail::check_implicit_conversion<U, DataType>();
    other.data = nullptr;
  }
};

template <typename Backend, typename DataType, int ndim>
template <int other_ndim>
TensorView<Backend, DataType, other_ndim> TensorViewBase<Backend, DataType, ndim>::to_static() {
  static_assert(other_ndim != DynamicDimensions,
                "Conversion to static only allowed for static shape");
  static_assert(ndim == other_ndim || ndim == DynamicDimensions, "Cannot convert to other ndim");
  // assert(other_ndim == dim() && "Cannot convert to other ndim");
  return {data, shape.template to_static<other_ndim>()};
}

/// @brief Non-owning list of Tensors in contigous memory.
///
/// Contains pointer to data, TensorListShape and offsets to all samples
/// For sample `i`, offsets[i] is an offset to first element and offsets[i+1] is an offset to
/// last + 1 element.
template <typename Backend, typename DataType, int sample_ndim = DynamicDimensions>
struct TensorListView;

template <typename Backend, typename DataType, int sample_ndim>
struct TensorListViewBase {
  using element_type = DataType;
  element_type *data = nullptr;
  TensorListShape<sample_ndim> shape;
  std::vector<ptrdiff_t> offsets;

  /// @brief Return non-ownin View to `sample`
  TensorView<Backend, DataType, sample_ndim> operator[](int sample) const {
    return {this->data + offsets[sample], shape[sample]};
  }

  template <int other_sample_ndim>
  TensorView<Backend, DataType, other_sample_ndim> tensor_view(int sample) const {
    static_assert(other_sample_ndim == sample_ndim || sample_ndim == DynamicDimensions
                  || other_sample_ndim == DynamicDimensions, "Cannot convert to other static ndim");
    return {this->data + offsets[sample], shape.template tensor_shape<other_sample_ndim>(sample)};
  }

  /// @brief Number of samples
  int size() const { return shape.size(); }
  int num_samples() const { return size(); }
  ptrdiff_t num_elements() const { return offsets.empty() ? 0 : offsets[size()]; }
  int sample_dim() const { return shape.sample_dim(); }

  /// @brief Update offsets after shape has been changed
  void update_offsets() {
    calculate_offsets(offsets, shape);
  }

  template <int other_sample_ndim>
  TensorListView<Backend, DataType, other_sample_ndim> to_static() {
    static_assert(other_sample_ndim != DynamicDimensions,
                  "Conversion to static only allowed for static shape");
    // Ofssets do not change
    return {data, shape.template to_static<other_sample_ndim>(), offsets};
  }

 protected:
  TensorListViewBase() = default;
  TensorListViewBase(const TensorListViewBase &) = default;
  TensorListViewBase(TensorListViewBase &&other)
      : data(other.data), shape(std::move(other.shape)), offsets(std::move(other.offsets)) {
    other.data = nullptr;
  }
  TensorListViewBase &operator=(const TensorListViewBase &) = default;
  TensorListViewBase &operator=(TensorListViewBase &&other) {
    data = other.data;
    other.data = nullptr;
    shape = std::move(other.shape);
    offsets = std::move(other.offsets);
    return *this;
  }
  TensorListViewBase(DataType *data, const TensorListShape<sample_ndim> &shapes)
      : data(data), shape(shapes), offsets(calculate_offsets(shape)) {}
  TensorListViewBase(DataType *data, TensorListShape<sample_ndim> &&shapes)
      : data(data), shape(std::move(shapes)), offsets(calculate_offsets(shape)) {}
  TensorListViewBase(DataType *data, const TensorListShape<sample_ndim> &shapes,
                     const std::vector<ptrdiff_t> &offsets)
      : data(data), shape(shapes), offsets(offsets) {}
  TensorListViewBase(DataType *data, TensorListShape<sample_ndim> &&shapes,
                     std::vector<ptrdiff_t> &&offsets)
      : data(data), shape(std::move(shapes)), offsets(std::move(offsets)) {}
};

template <typename Backend, typename DataType>
struct TensorListView<Backend, DataType, DynamicDimensions>
    : TensorListViewBase<Backend, DataType, DynamicDimensions> {
  using Base = TensorListViewBase<Backend, DataType, DynamicDimensions>;
  TensorListView() = default;
  TensorListView(const TensorListView &) = default;
  TensorListView(TensorListView &&) = default;
  TensorListView &operator=(const TensorListView &) = default;
  TensorListView &operator=(TensorListView &&) = default;
  TensorListView(DataType *data, const std::vector<TensorShape<DynamicDimensions>> &shapes)
      : Base(data, shapes) {}

  template <int other_sample_ndim>
  TensorListView(DataType *data, const TensorListShape<other_sample_ndim> &shape)
      : Base(data, shape) {}

  template <int other_sample_ndim>
  TensorListView(DataType *data, TensorListShape<other_sample_ndim> &&shape)
      : Base(data, std::move(shape)) {}

  TensorListView(DataType *data, const TensorListShape<DynamicDimensions> &shape,
                 const std::vector<ptrdiff_t> &offsets)
      : Base(data, shape, offsets) {}

  TensorListView(DataType *data, TensorListShape<DynamicDimensions> &&shape,
                 std::vector<ptrdiff_t> &&offsets)
      : Base(data, std::move(shape), std::move(offsets)) {}

  template <int other_sample_ndim, typename U>
  TensorListView(const TensorListView<Backend, U, other_sample_ndim> &other)
      : Base(other.data, other.shape, other.offsets) {
    detail::check_implicit_conversion<U, DataType>();
  }

  template <int other_sample_ndim, typename U>
  TensorListView(TensorListView<Backend, U, other_sample_ndim> &&other)
      : Base(other.data, std::move(other.shape), std::move(other.offsets)) {
    detail::check_implicit_conversion<U, DataType>();
    other.data = nullptr;
  }
};

template <typename Backend, typename DataType, int sample_ndim>
struct TensorListView : TensorListViewBase<Backend, DataType, sample_ndim> {
  using Base = TensorListViewBase<Backend, DataType, sample_ndim>;
  TensorListView() = default;
  TensorListView(const TensorListView &) = default;
  TensorListView(TensorListView &&) = default;
  TensorListView &operator=(const TensorListView &) = default;
  TensorListView &operator=(TensorListView &&) = default;
  TensorListView(DataType *data, const std::vector<TensorShape<sample_ndim>> &shapes)
      : Base(data, shapes) {}

  TensorListView(DataType *data, const TensorListShape<sample_ndim> &shape)
      : Base(data, shape) {}

  TensorListView(DataType *data, const TensorListShape<sample_ndim> &shape,
                 const std::vector<ptrdiff_t> &offsets)
      : Base(data, shape, offsets) {}

  TensorListView(DataType *data, TensorListShape<sample_ndim> &&shape,
                 std::vector<ptrdiff_t> &&offsets)
      : Base(data, std::move(shape), std::move(offsets)) {}


  template <typename U>
  TensorListView(const TensorListView<Backend, U, sample_ndim> &other)
      : Base(other.data, other.shape, other.offsets) {
    static_assert(sample_ndim == sample_ndim || sample_ndim == DynamicDimensions,
                  "Cannot change number of dimensions");
    detail::check_implicit_conversion<U, DataType>();
  }
  template <typename U>
  TensorListView(TensorListView<Backend, U, sample_ndim> &&other)
      : Base(other.data, std::move(other.shape), std::move(other.offsets)) {
    detail::check_implicit_conversion<U, DataType>();
    other.data = nullptr;
  }
};

struct StorageCPU;
struct StorageGPU;

/// @brief Wraps raw memory as a tensor
template <typename StorageBackend, int ndim, typename T>
TensorView<StorageBackend, T, ndim> make_tensor(T *data, TensorShape<ndim> shape) {
  return { data, std::move(shape) };
}

/// @brief Wraps raw memory as a tensor list
template <typename StorageBackend, int ndim, typename T>
TensorListView<StorageBackend, T, ndim> make_tensor_list(T *data, TensorListShape<ndim> shape) {
  return { data, std::move(shape) };
}

/// @brief Wraps CPU raw memory as a tensor
template <int ndim, typename T>
TensorView<StorageCPU, T, ndim> make_tensor_cpu(T *data, TensorShape<ndim> shape) {
  return { data, std::move(shape) };
}

/// @brief Wraps CPU raw memory as a tensor list
template <int ndim, typename T>
TensorListView<StorageCPU, T, ndim> make_tensor_list_cpu(T *data, TensorListShape<ndim> shape) {
  return { data, std::move(shape) };
}

/// @brief Wraps GPU raw memory as a tensor
template <int ndim, typename T>
TensorView<StorageGPU, T, ndim> make_tensor_gpu(T *data, TensorShape<ndim> shape) {
  return { data, std::move(shape) };
}

/// @brief Wraps GPU raw memory as a tensor list
template <int ndim, typename T>
TensorListView<StorageGPU, T, ndim> make_tensor_list_gpu(T *data, TensorListShape<ndim> shape) {
  return { data, std::move(shape) };
}


/// @brief Get a subtensor by slicing along outermost dimension at position `pos`
///
/// @details Produces tensor, for which number of dimensions is reduced by 1.
/// Removed dimension is outer-most (e.g. for shape {3,2,4,6} produces {2,4,6}).
/// Data inside the tensor is extracted according to provided index.
/// Data is not copied.
///
/// Example:
/// tv.data = [[1, 2, 3], [4, 5, 6]]       (shape: [2, 3])
/// oust_dimension(tv, 1) -> [4, 5, 6]     (shape: [3])
///
/// @param source Source TensorView
/// @param idx Index inside dimension, along which data is extracted
/// @return TensorView with reduced dimensionality
template<typename StorageBackend, typename DataType, int ndims>
TensorView<StorageBackend, DataType, ndims - 1>
subtensor(TensorView<StorageBackend, DataType, ndims> source, int64_t pos) {
  TensorShape<ndims - 1> shape = source.shape.template last<ndims - 1>();
  DataType *data = source.data + pos * volume(shape);
  return make_tensor<StorageBackend>(data, shape);
}


/// @brief Overload for Dynamic TensorView
template<typename StorageBackend, typename DataType>
TensorView<StorageBackend, DataType, DynamicDimensions>
subtensor(TensorView<StorageBackend, DataType, DynamicDimensions> source, int64_t pos) {
  auto shape = source.shape.last(source.dim() - 1);
  DataType *data = source.data + pos * volume(shape);
  return make_tensor<StorageBackend>(data, std::move(shape));
}

}  // namespace kernels

template <typename Backend, typename T, int ndim>
struct element_type<kernels::TensorView<Backend, T, ndim>> {
  using type = T;
};

template <typename Backend, typename T, int ndim>
struct element_type<kernels::TensorListView<Backend, T, ndim>> {
  using type = T;
};

}  // namespace dali

#endif  // DALI_KERNELS_TENSOR_VIEW_H_
