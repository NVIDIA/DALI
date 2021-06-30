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

#ifndef DALI_CORE_TENSOR_VIEW_H_
#define DALI_CORE_TENSOR_VIEW_H_

#include <type_traits>
#include <utility>
#include <vector>
#include <stdexcept>
#include "dali/core/tensor_shape.h"

namespace dali {

namespace detail {

template <typename From, typename To>
struct check_implicit_conversion {
  static_assert(std::is_convertible<From*, To*>::value, "Conversion impossible");
  static_assert(std::is_same<
    std::remove_cv_t<From>,
    std::remove_cv_t<To>>::value,
    "Implicit conversion can only change CV qualifiers");
};

}  // namespace detail

template <typename Shape, typename Position>
bool ContainsCoords(const Shape &shape, const Position &pos) {
  const int shape_dim = size(shape);
  const int pos_dim = size(pos);
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

/**
 * @brief Calculates flat index of a given element in the tensor
 * @remarks If pos has fewer dimensions than shape, the remaining offsets are assumed to be 0
 */
DALI_NO_EXEC_CHECK
template <typename Shape, typename Position>
DALI_HOST_DEV if_array_like<Position, ptrdiff_t> CalcOffset(const Shape &shape,
                                                            const Position &pos) {
  ptrdiff_t ofs = pos[0];
  const int pos_dim = size(pos);
  const int shape_dim = size(shape);
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

/**
 * @brief Calculates the offset to a slice of the tensor
 */
template <typename Shape>
DALI_HOST_DEV ptrdiff_t CalcOffset(const Shape &shape, const ptrdiff_t &index) {
  ptrdiff_t ofs = index;
  const int shape_dim = size(shape);
  for (int i = 1; i < shape_dim; i++) {
    ofs *= shape[i];
  }
  return ofs;
}

struct EmptyBackendTag {};

/**
 * @brief Non-owning wrapper for Tensor, containing typed pointer to data and TensorShape
 *
 * @tparam Backend
 * @tparam DataType
 * @tparam ndim either static for ndim >= 0 or DynamicDimensions
 */
template <typename Backend, typename DataType, int ndim = DynamicDimensions>
struct TensorView;

template <typename Backend, typename DataType, int ndim>
struct TensorViewBase {
  using element_type = DataType;
  int dim() const { return shape.size(); }

  /**
   * @brief Utility to calculate pointer to element at given coordinates
   */
  template <typename... Indices>
  DataType *operator()(int64_t idx0, Indices &&... idx) const {
    return data + CalcOffset(shape, std::array<ptrdiff_t, sizeof...(Indices) + 1>{
                                        idx0, (ptrdiff_t{idx})...});
  }

  /**
   * @brief Utility to calculate pointer to element at given coordinates
   */
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

/**
 * @brief Dynamic TensorView can be constructed from any Static TensorView
 */
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

/**
 * @brief Non-owning list of Tensors.
 *
 * Contains TensorListShape and pointers to the beginning of each Tensor.
 * For sample `i`, offsets[i] is an offset to first element and offsets[i+1] is an offset to
 * last + 1 element.
 * Shape and pointers are stored in contiguous memory for improved data locality and reduced
 * number of allocations.
 */
template <typename Backend, typename DataType, int sample_ndim = DynamicDimensions>
struct TensorListView;

template <typename Backend, typename DataType, int sample_ndim>
struct TensorListViewBase {
  using element_type = DataType;

  /**
   * @brief Return non-owning View to sample at specified index
   */
  TensorView<Backend, DataType, sample_ndim> operator[](int sample) const {
    return { tensor_data(sample), tensor_shape(sample) };
  }

  template <int other_sample_ndim>
  TensorView<Backend, DataType, other_sample_ndim> tensor_view(int sample) const {
    static_assert(other_sample_ndim == sample_ndim || sample_ndim == DynamicDimensions
                  || other_sample_ndim == DynamicDimensions, "Cannot convert to other static ndim");
    return { data[sample], shape.template tensor_shape<other_sample_ndim>(sample)};
  }

  /**
   * @brief Number of samples
   */
  int size() const noexcept { return shape.size(); }
  int num_samples() const noexcept { return size(); }
  ptrdiff_t num_elements() const {
    return shape.num_elements();
  }
  int sample_dim() const { return shape.sample_dim(); }

  void resize(int num_samples) {
    shape.resize(num_samples);
    data.resize(num_samples);
  }

  void resize(int num_samples, int dim) {
    shape.resize(num_samples, dim);
    data.resize(num_samples);
  }

  bool empty() const {
    return data.empty();
  }

  explicit operator bool() const {
    return !empty();
  }

  template <int other_sample_ndim>
  TensorListView<Backend, DataType, other_sample_ndim> to_static() const & {
    static_assert(other_sample_ndim != DynamicDimensions,
                  "Conversion to static only allowed for static shape");
    return { data, shape.template to_static<other_sample_ndim>() };
  }

  template <int other_sample_ndim>
  TensorListView<Backend, DataType, other_sample_ndim> to_static() && {
    static_assert(other_sample_ndim != DynamicDimensions,
                  "Conversion to static only allowed for static shape");
    return { std::move(data), std::move(shape).template to_static<other_sample_ndim>() };
  }

  DataType *tensor_data(int sample) const {
    return data[sample];
  }

  DataType *&tensor_data(int sample) {
    return data[sample];
  }

  template <int output_dim = sample_ndim>
  TensorShape<output_dim> tensor_shape(int sample) const {
    return shape.template tensor_shape<output_dim>(sample);
  }

  auto tensor_shape_span(int sample) {
    return shape.tensor_shape_span(sample);
  }

  auto tensor_shape_span(int sample) const {
    return shape.tensor_shape_span(sample);
  }

  /**
   * @brief Sets a base pointer for the entire list, assuming contiguous layout of tensors.
   *
   * The data pointerd of individual tensors in the lists are calculated assuming that the
   * tensors are stored sequentially and without gaps.
   *
   * If the tensor list was populated using this function, it is guaranteed that
   * @ref is_contiguous returns true.
   */
  template <typename U>
  void set_contiguous_data(U *data) {
    detail::check_implicit_conversion<U, DataType>();
    calculate_pointers(this->data, data, shape);
  }

  /**
   * @brief Checks whether the tensors in the list are packed sequentially, without gaps.
   *
   * The function checks if the data pointer to a next sample immediately follows the last
   * element of the previous sample. There's no constraint on the shapes of the samples, however.
   * To determine whether the tensor list can be reduced to a tensor, see @ref is_tensor.
   */
  bool is_contiguous() const {
    if (num_samples() < 2)
      return true;
    auto *ptr = data[0];
    for (int i = 1; i < num_samples(); i++) {
      auto *next = ptr + volume(tensor_shape_span(i - 1));
      if (data[i] != next)
        return false;
      ptr = next;
    }
    return true;
  }

  /**
   * @brief Checks whether the tensor list can be viewed as a tensor, with samples
   *        as outermost dimension.
   *
   * A the tensor list can be viewed as a tensor with an extra dimesnions (samples)
   * if all the tensors in that list are of the same shape and the storage is contiguous.
   * Some algorithms may take advantage of this and choose an optimized code path.
   *
   * @return *true*  if the shape is uniform and the storage is contiguous;
   *         *false* otherwise
   */
  bool is_tensor() const {
    return is_uniform(shape) && is_contiguous();
  }

  using data_pointers_t = std::vector<element_type*>;
  TensorListShape<sample_ndim> shape;
  data_pointers_t data;

 protected:
  TensorListViewBase() = default;
  TensorListViewBase(const TensorListViewBase &) = default;
  TensorListViewBase(TensorListViewBase &&other) = default;
  TensorListViewBase &operator=(const TensorListViewBase &) = default;
  TensorListViewBase &operator=(TensorListViewBase &&other) = default;

  /**
   * @brief Constructs a tensor list without specific memory
   *
   * The shape is copied from `shape` parameter and the `data`
   * vector is resized to num_samples and filled with null pointers.
   */
  TensorListViewBase(const TensorListShape<sample_ndim> &shape)  // NOLINT
      : shape(shape)
      , data(this->num_samples(), nullptr) {}

  /**
   * @brief Constructs a tensor list without specific memory
   *
   * The shape is taken from `shape` parameter and the `data`
   * vector is resized to num_samples and filled with null pointers.
   */
  TensorListViewBase(TensorListShape<sample_ndim> &&shape)  // NOLINT
      : shape(std::move(shape))
      , data(this->num_samples(), nullptr) {}

  /**
   * @brief Constructs a tensor list from non-contiguous memory
   */
  TensorListViewBase(const data_pointers_t &data, const TensorListShape<sample_ndim> &shape)
      : shape(shape)
      , data(data) {}
  /**
   * @brief Constructs a tensor list from non-contiguous memory
   */
  TensorListViewBase(data_pointers_t &&data, TensorListShape<sample_ndim> &&shape)
      : shape(std::move(shape))
      , data(std::move(data)) {}


  /**
   * @brief Constructs a tensor list from non-contiguous memory
   */
  template <typename U>
  TensorListViewBase(const std::vector<U*> &data, const TensorListShape<sample_ndim> &shape)
      : shape(shape)
      , data(reinterpret_cast<const data_pointers_t&>(data)) {
    detail::check_implicit_conversion<U, DataType>();
  }

  /**
   * @brief Constructs a tensor list from non-contiguous memory
   */
  template <typename U>
  TensorListViewBase(std::vector<U*> &&data, TensorListShape<sample_ndim> &&shape)
      : shape(std::move(shape))
      , data(std::move(reinterpret_cast<data_pointers_t&>(data))) {
    detail::check_implicit_conversion<U, DataType>();
  }

  /**
   * @brief Constructs a tensor list from non-contiguous memory
   */
  TensorListViewBase(DataType *const *data, const TensorListShape<sample_ndim> &shape)
      : shape(shape)
      , data(data, data + this->shape.num_samples()) {}
  /**
   * @brief Constructs a tensor list from non-contiguous memory
   */
  TensorListViewBase(DataType *const *data, TensorListShape<sample_ndim> &&shape)
      : shape(std::move(shape))
      , data(data, data + this->shape.num_samples()) {}

  /**
   * @brief Constructs a tensor list from contiguous memory
   */
  TensorListViewBase(DataType *data, const TensorListShape<sample_ndim> &shape)
      : shape(shape) {
    calculate_pointers(this->data, data, this->shape);
  }
  /**
   * @brief Constructs a tensor list from contiguous memory
   */
  TensorListViewBase(DataType *data, TensorListShape<sample_ndim> &&shape)
      : shape(std::move(shape)) {
    calculate_pointers(this->data, data, this->shape);
  }
};

template <typename Backend, typename DataType>
struct TensorListView<Backend, DataType, DynamicDimensions>
    : TensorListViewBase<Backend, DataType, DynamicDimensions> {
  using Base = TensorListViewBase<Backend, DataType, DynamicDimensions>;
  using typename Base::data_pointers_t;
  TensorListView() = default;
  TensorListView(const TensorListView &) = default;
  TensorListView(TensorListView &&) = default;
  TensorListView &operator=(const TensorListView &) = default;
  TensorListView &operator=(TensorListView &&) = default;

  //@{
  /** @brief  Construction from contiguous memory */

  TensorListView(DataType *data, const std::vector<TensorShape<DynamicDimensions>> &shapes)
      : Base(data, shapes) {}

  template <int other_sample_ndim>
  TensorListView(DataType *data, const TensorListShape<other_sample_ndim> &shape)
      : Base(data, shape) {}

  template <int other_sample_ndim>
  TensorListView(DataType *data, TensorListShape<other_sample_ndim> &&shape)
      : Base(data, std::move(shape)) {}

  //@}

  //@{
  /** @brief Construction from non-contiguous memory */

  TensorListView(DataType *const *data, const std::vector<TensorShape<DynamicDimensions>> &shapes)
      : Base(data, shapes) {}

  template <int other_sample_ndim>
  TensorListView(DataType *const *data, const TensorListShape<other_sample_ndim> &shape)
      : Base(data, shape) {}

  template <int other_sample_ndim>
  TensorListView(DataType *const *data, TensorListShape<other_sample_ndim> &&shape)
      : Base(data, std::move(shape)) {}

  template <int other_sample_ndim>
  TensorListView(const data_pointers_t &data, const TensorListShape<other_sample_ndim> &shape)
      : Base(data, shape) {}

  template <int other_sample_ndim>
  TensorListView(data_pointers_t &&data, TensorListShape<other_sample_ndim> &&shape)
      : Base(std::move(data), std::move(shape)) {}

  //@}

  //@{
  /** @brief Implicit conversion */

  template <int other_sample_ndim, typename U>
  TensorListView(const TensorListView<Backend, U, other_sample_ndim> &other)
      : Base(other.data, other.shape) {
    detail::check_implicit_conversion<U, DataType>();
  }

  template <int other_sample_ndim, typename U>
  TensorListView(TensorListView<Backend, U, other_sample_ndim> &&other)
      : Base(std::move(other.data), std::move(other.shape)) {
    detail::check_implicit_conversion<U, DataType>();
  }

  //@}
};

template <typename Backend, typename DataType, int sample_ndim>
struct TensorListView : TensorListViewBase<Backend, DataType, sample_ndim> {
  using Base = TensorListViewBase<Backend, DataType, sample_ndim>;
  using typename Base::data_pointers_t;
  TensorListView() = default;
  TensorListView(const TensorListView &) = default;
  TensorListView(TensorListView &&) = default;
  TensorListView &operator=(const TensorListView &) = default;
  TensorListView &operator=(TensorListView &&) = default;

  //@{
  /** @brief Construction from contiguous memory */

  TensorListView(std::nullptr_t, const std::vector<TensorShape<sample_ndim>> &shapes)
      : Base(TensorListShape<sample_ndim>(shapes)) {}

  template <int other_sample_ndim>
  TensorListView(std::nullptr_t, const TensorListShape<other_sample_ndim> &shape)
      : Base(shape) {}

  template <int other_sample_ndim>
  TensorListView(std::nullptr_t, TensorListShape<other_sample_ndim> &&shape)
      : Base(std::move(shape)) {}

  //@}

  //@{
  /** @brief Construction from contiguous memory */

  TensorListView(DataType *data, const std::vector<TensorShape<sample_ndim>> &shapes)
      : Base(data, TensorListShape<sample_ndim>(shapes)) {}

  template <int other_sample_ndim>
  TensorListView(DataType *data, const TensorListShape<other_sample_ndim> &shape)
      : Base(data, shape) {}

  template <int other_sample_ndim>
  TensorListView(DataType *data, TensorListShape<other_sample_ndim> &&shape)
      : Base(data, std::move(shape)) {}

  //@}

  //@{
  /** @brief Construction from non-contiguous memory */

  TensorListView(DataType *const *data, const std::vector<TensorShape<sample_ndim>> &shapes)
      : Base(data, shapes) {}

  template <int other_sample_ndim>
  TensorListView(DataType *const *data, const TensorListShape<other_sample_ndim> &shape)
      : Base(data, shape) {}

  template <int other_sample_ndim>
  TensorListView(DataType *const *data, TensorListShape<other_sample_ndim> &&shape)
      : Base(data, std::move(shape)) {}

  TensorListView(const data_pointers_t &data, const TensorListShape<sample_ndim> &shape)
      : Base(data, shape) {}

  TensorListView(data_pointers_t &&data, TensorListShape<sample_ndim> &&shape)
      : Base(std::move(data), std::move(shape)) {}

  //@}

  //@{
  /** @brief Implicit conversion */

  template <typename U, int other_sample_ndim>
  TensorListView(const TensorListView<Backend, U, other_sample_ndim> &other)
      : Base(other.data, other.shape) {
    static_assert(sample_ndim == other_sample_ndim || sample_ndim == DynamicDimensions,
                  "Cannot change number of dimensions");
    detail::check_implicit_conversion<U, DataType>();
  }
  template <typename U>
  TensorListView(TensorListView<Backend, U, sample_ndim> &&other)
      : Base(std::move(other.data), std::move(other.shape)) {
    detail::check_implicit_conversion<U, DataType>();
  }

  //@}
};

struct StorageCPU;
struct StorageGPU;

/**
 * @brief Wraps raw memory as a tensor
 */
template <typename StorageBackend, int ndim, typename T>
TensorView<StorageBackend, T, ndim> make_tensor(T *data, TensorShape<ndim> shape) {
  return { data, std::move(shape) };
}

/**
 * @brief Wraps raw memory as a tensor list
 */
template <typename StorageBackend, int ndim, typename T>
TensorListView<StorageBackend, T, ndim> make_tensor_list(T *data, TensorListShape<ndim> shape) {
  return { data, std::move(shape) };
}

/**
 * @brief Wraps raw memory as a tensor list
 */
template <typename StorageBackend, int ndim, typename T>
TensorListView<StorageBackend, T, ndim> make_tensor_list(T *const *data,
                                                         TensorListShape<ndim> shape) {
  return { data, std::move(shape) };
}

/**
 * @brief Wraps CPU raw memory as a tensor
 */
template <int ndim, typename T>
TensorView<StorageCPU, T, ndim> make_tensor_cpu(T *data, TensorShape<ndim> shape) {
  return { data, std::move(shape) };
}

/**
 * @brief Wraps contiguous CPU memory as a tensor list
 */
template <int ndim, typename T>
TensorListView<StorageCPU, T, ndim> make_tensor_list_cpu(T *data, TensorListShape<ndim> shape) {
  return { data, std::move(shape) };
}

/**
 * @brief Wraps CPU raw memory as a tensor list
 */
template <int ndim, typename T>
TensorListView<StorageCPU, T, ndim> make_tensor_list_cpu(T *const *data,
                                                         TensorListShape<ndim> shape) {
  return { data, std::move(shape) };
}

/**
 * @brief Wraps GPU raw memory as a tensor
 */
template <int ndim, typename T>
TensorView<StorageGPU, T, ndim> make_tensor_gpu(T *data, TensorShape<ndim> shape) {
  return { data, std::move(shape) };
}

/**
 * @brief Wraps contiguous GPU memory as a tensor list
 */
template <int ndim, typename T>
TensorListView<StorageGPU, T, ndim> make_tensor_list_gpu(T *data, TensorListShape<ndim> shape) {
  return { data, std::move(shape) };
}

/**
 * @brief Wraps GPU raw memory as a tensor list
 */
template <int ndim, typename T>
TensorListView<StorageGPU, T, ndim> make_tensor_list_gpu(T *const *data,
                                                         TensorListShape<ndim> shape) {
  return { data, std::move(shape) };
}

/**
 * @{
 * @brief Get a subtensor by slicing along the outermost dimension at position `pos`
 *
 * @details Produces a tensor with the outermost extent removed,
 * (e.g. for shape {3,2,4,6} produces {2,4,6}).
 * The data pointer in the new tensor points to the subtensor at the index `pos`.
 * No copy or allocation (except possibly the shape) occurs.
 *
 * Example:
 * tv.data = [[1, 2, 3], [4, 5, 6]]       (shape: [2, 3])
 * oust_dimension(tv, 1) -> [4, 5, 6]     (shape: [3])
 *
 * @param source Source TensorView
 * @param idx Outermost index
 * @return The tensor slice
 */
template <typename StorageBackend, typename DataType, int ndim>
TensorView<StorageBackend, DataType, ndim - 1>
subtensor(TensorView<StorageBackend, DataType, ndim> source, int64_t pos) {
  TensorShape<ndim - 1> shape = source.shape.template last<ndim - 1>();
  DataType *data = source.data + pos * volume(shape);
  return make_tensor<StorageBackend>(data, shape);
}


template <typename StorageBackend, typename DataType>
TensorView<StorageBackend, DataType, DynamicDimensions>
subtensor(TensorView<StorageBackend, DataType, DynamicDimensions> source, int64_t pos) {
  auto shape = source.shape.last(source.dim() - 1);
  DataType *data = source.data + pos * volume(shape);
  return make_tensor<StorageBackend>(data, std::move(shape));
}
/**
 * @}
 */

/**
 * @brief Merges a dimension with the next one
 *
 * @details The output tensor represents the same data, but the shape has two dimensions
 * collapsed into one, e.g. a tensor with shape
 * [2, 3, 4, 5]
 * after a call to collapse_dim(tensor, 1) would have a shape:
 * [2, 12, 5]
 *
 * @param dim_idx - the dimension to drop; must be >= 0 and < tv.dim() - 1
 * @param rv - input TensorView
 * @remarks The `dim_idx` must not be the innermost dimension or the result is undefined.
 */
template <typename StorageBackend, typename DataType, int ndim>
auto collapse_dim(const TensorView<StorageBackend, DataType, ndim> &tv, int dim_idx) {
  return make_tensor<StorageBackend>(tv.data, collapse_dim(tv.shape, dim_idx));
}

/**
 * @brief Retrieves a sample range from a tensor list
 * @param input      input list
 * @param out_slice  output list
 * @param begin      index of the first sample to include in the subrange
 * @param end        index one past the last sample to include in the subrange
 * @param step       stride between consecutive source samples
 */
template <typename StorageBackend, typename DataType, int out_ndim, int ndim>
void sample_range(TensorListView<StorageBackend, DataType, out_ndim> &out_slice,
    const TensorListView<StorageBackend, DataType, ndim> &input, int begin, int end, int step = 1) {
  detail::check_compatible_ndim<out_ndim, ndim>();
  assert(begin >= 0 && begin <= input.num_samples());
  assert(end >= begin && end <= input.num_samples());
  sample_range(out_slice.shape, input.shape, begin, end, step);
  out_slice.data.resize(out_slice.shape.num_samples());
  for (int i = begin, j = 0; i < end; i += step, j++)
    out_slice.data[j] = input.data[i];
}


/**
 * @brief Retrieves a sample range from a tensor list
 * @param input      input list
 * @param begin      index of the first sample to include in the subrange
 * @param end        index one past the last sample to include in the subrange
 * @param step       stride between consecutive source samples
 * @return `TensorListView<out_ndim>` consisting of samples at indices `begin` to `end` - 1
 */
template <int out_ndim = InferDimensions, typename StorageBackend, typename DataType, int ndim,
  int output_ndim = (out_ndim == InferDimensions ? ndim : out_ndim)>
TensorListView<StorageBackend, DataType, output_ndim> sample_range(
    const TensorListView<StorageBackend, DataType, ndim> &input, int begin, int end, int step = 1) {
  TensorListView<StorageBackend, DataType, output_ndim> out_slice;
  sample_range(out_slice, input, begin, end, step);
  return out_slice;
}


/**
 * @brief Uses existing data and a new shape and type to construct a new TensorListView.
 *
 * The function combines existing list with a new shape to build a new TensorListView,
 * possibly with different element type.
 * Non-contiguous input samples must not contribute to one output sample;
 * merging contiguous samples and splitting is still possible.
 *
 * @tparam U    new element type
 * @param list  original tensor list
 * @param shape the desired shape
 * @param check if true, exception is thrown when the list cannot be reshaped
 */
template <typename U, int out_dim, typename Storage, typename T, int in_dim>
TensorListView<Storage, U, out_dim> reinterpret(
      const TensorListView<Storage, T, in_dim> &list,
      TensorListShape<out_dim> shape,
      bool check = false) {
  if (check && shape.num_elements() * sizeof(U) != list.num_elements() * sizeof(T))
    throw std::logic_error("Attempt to reshape a TensorListView to a different total size");
  assert(shape.num_elements() * sizeof(U) == list.num_elements() * sizeof(T));

  int i = -1;  // start at index -1 - we'll increase it anyway
  int M = list.num_samples();
  int N = shape.num_samples();

  if (!N)
    return make_tensor_list<Storage, out_dim>(static_cast<U*>(nullptr), std::move(shape));

  ptrdiff_t in_remaining = 0;
  uintptr_t ptr = reinterpret_cast<uintptr_t>(list.data[0]);

  TensorListView<Storage, U, out_dim> out_list;
  out_list.shape = std::move(shape);
  out_list.data.resize(N);

  for (int o = 0; o < N; o++) {
    ptrdiff_t out_remaining = volume(out_list.shape.tensor_shape_span(o)) * sizeof(U);
    out_list.data[o] = reinterpret_cast<U *>(ptr);
    bool first_chunk_in_sample = true;
    while (out_remaining > 0) {
      if (in_remaining > 0) {
        ptrdiff_t to_add = out_remaining < in_remaining ? out_remaining : in_remaining;
        out_remaining -= to_add;
        in_remaining -= to_add;
        ptr += to_add;
      } else {
        i++;
        assert(i < M);
        uintptr_t next_ptr = reinterpret_cast<uintptr_t>(list.data[i]);
        in_remaining = volume(list.shape.tensor_shape_span(i)) * sizeof(T);
        if (!in_remaining)
          continue;  // empty input sample - skip and don't reset the first chunk flag
        if (first_chunk_in_sample) {
          ptr = next_ptr;
          out_list.data[o] = reinterpret_cast<U *>(ptr);
        } else {
          if (check && next_ptr != ptr)
            throw std::logic_error("Cannot merge non-contiguous samples");
          assert(next_ptr == ptr);
          // no need to update ptr, the value is good
        }
      }
      first_chunk_in_sample = false;
    }
  }
  assert(in_remaining == 0);
  return out_list;
}

/**
 * @brief Uses existing data and a new shape to construct a new TensorListView.
 *
 * The function combines existing list with a new shape to build a new TensorListView.
 * Non-contiguous input samples must not contribute to one output sample;
 * merging contiguous samples and splitting is still possible.
 *
 * @param list  original tensor list
 * @param shape the desired shape
 * @param check if true, exception is thrown when the list cannot be reshaped
 */
template <int out_dim, typename Storage, typename T, int in_dim>
TensorListView<Storage, T, out_dim> reshape(
      const TensorListView<Storage, T, in_dim> &list,
      TensorListShape<out_dim> shape,
      bool check = false) {
  return reinterpret<T, out_dim>(list, shape, check);
}

template <typename Backend, typename T, int ndim>
struct element_type<TensorView<Backend, T, ndim>> {
  using type = T;
};

template <typename Backend, typename T, int ndim>
struct element_type<TensorListView<Backend, T, ndim>> {
  using type = T;
};

}  // namespace dali

#endif  // DALI_CORE_TENSOR_VIEW_H_
