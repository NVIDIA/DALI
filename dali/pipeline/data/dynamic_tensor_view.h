// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_DATA_DYNAMIC_TENSOR_VIEW_H_
#define DALI_PIPELINE_DATA_DYNAMIC_TENSOR_VIEW_H_

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_view.h"
#include "dali/pipeline/data/types.h"

namespace dali {

/**
 * @brief Tag type indicating that the TensorView is keeping the type information in runtime.
 *
 * It can be converted to the statically-typed TensorView using `view<T, ndim>()` function
 * or by `TensorView::to_static_type<T, ndim>()` member function.
 */
struct DynamicType {
  DynamicType() = delete;
};

template <typename Backend, typename DataType, int ndim = DynamicDimensions>
struct DynamicTensorViewBase {
  static_assert(std::is_same<std::remove_const_t<DataType>, void>::value,
                "The underlying type must either be void or const void");
  using element_type = DataType;
  int dim() const {
    return shape.sample_dim();
  }
  DALIDataType type() const {
    return type_id;
  }

  ptrdiff_t num_elements() const {
    return shape.num_elements();
  }

  /**
   * @brief Utility to calculate pointer to element at given coordinates
   */
  template <typename... Indices>
  DataType *operator()(int64_t idx0, Indices &&...idx) const {
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

  DataType *data = nullptr;
  TensorShape<ndim> shape = {};
  DALIDataType type_id = DALI_NO_TYPE;

 protected:
  DynamicTensorViewBase() = default;
  DynamicTensorViewBase(const DynamicTensorViewBase &) = default;
  DynamicTensorViewBase(DataType *data, const TensorShape<ndim> &shape, DALIDataType type_id)
      : data(data), shape(shape), type_id(type_id) {}
  DynamicTensorViewBase(DataType *data, TensorShape<ndim> &&shape, DALIDataType type_id)
      : data(data), shape(std::move(shape)), type_id(type_id) {}
};

template <typename Backend, int ndim = DynamicDimensions>
struct DynamicTensorView : DynamicTensorViewBase<Backend, void, ndim> {
  using Base = DynamicTensorViewBase<Backend, void, ndim>;

  DynamicTensorView() = default;

  /**
   * @name Construct the view inferring the type_id from the pointer value.
   */
  // @{
  template <typename T, typename = std::enable_if_t<!std::is_const<T>::value>>
  DynamicTensorView(T *data, const TensorShape<ndim> &shape)
      : Base(data, shape, TypeTable::GetTypeId<T>()) {}

  template <typename T, typename = std::enable_if_t<!std::is_const<T>::value>>
  DynamicTensorView(T *data, TensorShape<ndim> &&shape)
      : Base(data, std::move(shape), TypeTable::GetTypeId<T>()) {}

  template <typename T, int other_ndim, typename = std::enable_if_t<!std::is_const<T>::value>>
  DynamicTensorView(T *data, const TensorShape<other_ndim> &shape)
      : Base(data, shape, TypeTable::GetTypeId<T>()) {
    // TODO(klecki): The tensor shape goes through a runtime check for some reason, so we
    // temporarily plug it here before we evaluate TensorShape fix
    detail::check_compatible_ndim<ndim, other_ndim>();
  }

  template <typename T, int other_ndim, typename = std::enable_if_t<!std::is_const<T>::value>>
  DynamicTensorView(T *data, TensorShape<other_ndim> &&shape)
      : Base(data, std::move(shape), TypeTable::GetTypeId<T>()) {
    detail::check_compatible_ndim<ndim, other_ndim>();
  }
  // @}

  /**
   * @name Construct the view with explicitly provided type_id.
   */
  // @{
  DynamicTensorView(void *data, const TensorShape<ndim> &shape, DALIDataType type_id)
      : Base(data, shape, type_id) {}

  DynamicTensorView(void *data, TensorShape<ndim> &&shape, DALIDataType type_id)
      : Base(data, std::move(shape), type_id) {}

  template <int other_ndim>
  DynamicTensorView(void *data, const TensorShape<other_ndim> &shape, DALIDataType type_id)
      : Base(data, shape, type_id) {
    detail::check_compatible_ndim<ndim, other_ndim>();
  }

  template <int other_ndim>
  DynamicTensorView(void *data, TensorShape<other_ndim> &&shape, DALIDataType type_id)
      : Base(data, std::move(shape), type_id) {
    detail::check_compatible_ndim<ndim, other_ndim>();
  }
  // @}

  /**
   * @name nullptr overloads with DALI_NO_TYPE
   */
  // @{
  DynamicTensorView(std::nullptr_t *, const TensorShape<ndim> &shape)
      : Base(nullptr, shape, DALI_NO_TYPE) {}

  DynamicTensorView(std::nullptr_t *, TensorShape<ndim> &&shape)
      : Base(nullptr, std::move(shape), DALI_NO_TYPE) {}

  template <int other_ndim>
  DynamicTensorView(std::nullptr_t *, const TensorShape<other_ndim> &shape)
      : Base(nullptr, shape, DALI_NO_TYPE) {
    detail::check_compatible_ndim<ndim, other_ndim>();
  }

  template <int other_ndim>
  DynamicTensorView(std::nullptr_t *, TensorShape<other_ndim> &&shape)
      : Base(nullptr, std::move(shape), DALI_NO_TYPE) {
    detail::check_compatible_ndim<ndim, other_ndim>();
  }
  // @}

  /**
   * @name Copy and move constructor and assignment ops.
   */
  // @{
  template <int other_ndim>
  explicit DynamicTensorView(const DynamicTensorView<Backend, other_ndim> &other)
      : Base(other.data, other.shape, other.type_id) {
    detail::check_compatible_ndim<ndim, other_ndim>();
  }

  template <int other_ndim>
  DynamicTensorView &operator=(const DynamicTensorView<Backend, other_ndim> &other) {
    detail::check_compatible_ndim<ndim, other_ndim>();
    this->data = other.data;
    this->shape = other.shape;
    this->type_id = other.type_id;
    return *this;
  }

  template <int other_ndim>
  explicit DynamicTensorView(DynamicTensorView<Backend, other_ndim> &&other) {
    detail::check_compatible_ndim<ndim, other_ndim>();
    this->data = other.data;
    other.data = nullptr;
    this->shape = std::move(other.shape);
    this->type_id = other.type_id;
    other.type_id = DALI_NO_TYPE;
  }

  template <int other_ndim>
  DynamicTensorView &operator=(DynamicTensorView<Backend, other_ndim> &&other) {
    detail::check_compatible_ndim<ndim, other_ndim>();
    this->data = other.data;
    other.data = nullptr;
    this->shape = std::move(other.shape);
    this->type_id = other.type_id;
    other.type_id = DALI_NO_TYPE;
    return *this;
  }
  // @}

  /**
   * @name Converters from static TensorView
   */
  // @{
  template <
      typename T, int other_ndim,
      typename = std::enable_if_t<!std::is_const<T>::value && !std::is_same<T, DynamicType>::value>>
  explicit DynamicTensorView(const TensorView<Backend, T, other_ndim> &other) {
    detail::check_compatible_ndim<ndim, other_ndim>();
    this->data = other.data;
    this->shape = other.shape;
    this->type_id = TypeTable::GetTypeId<T>();
  }

  template <
      typename T, int other_ndim,
      typename = std::enable_if_t<!std::is_const<T>::value && !std::is_same<T, DynamicType>::value>>
  explicit DynamicTensorView(TensorView<Backend, T, other_ndim> &&other) {
    detail::check_compatible_ndim<ndim, other_ndim>();
    this->data = other.data;
    other.data = nullptr;
    this->shape = std::move(other.shape);
    this->type_id = TypeTable::GetTypeId<T>();
  }

  template <
      typename T, int other_ndim,
      typename = std::enable_if_t<!std::is_const<T>::value && !std::is_same<T, DynamicType>::value>>
  DynamicTensorView &operator=(const TensorView<Backend, T, other_ndim> &other) {
    detail::check_compatible_ndim<ndim, other_ndim>();
    this->data = other.data;
    this->shape = other.shape;
    this->type_id = TypeTable::GetTypeId<T>();
    return *this;
  }

  template <
      typename T, int other_ndim,
      typename = std::enable_if_t<!std::is_const<T>::value && !std::is_same<T, DynamicType>::value>>
  DynamicTensorView &operator=(TensorView<Backend, T, other_ndim> &&other) {
    detail::check_compatible_ndim<ndim, other_ndim>();
    this->data = other.data;
    other.data = nullptr;
    this->shape = std::move(other.shape);
    this->type_id = TypeTable::GetTypeId<T>();
    return *this;
  }
  // @}


  /**
   * @name Explicitly deleted constructors disallowing passing a pointer to const.
   *
   * Listing all the variants here, and blocking others with SFINAE allows the compiler
   * to state that such constructor is deleted rather than trying to instantiate the type2id
   * trait with const type and failing miserably with supper long message.
   *
   * If you see any of the constructors below being used, it means that you tired to pass pointer to
   * const to a non-const view container, which is not allowed.
   */
  // @{
  template <typename T>
  DynamicTensorView(const T *data, const TensorShape<ndim> &shape) = delete;
  template <typename T>
  DynamicTensorView(const T *data, const TensorShape<ndim> &shape, DALIDataType type_id) = delete;
  template <typename T, int other_ndim>
  DynamicTensorView(const T *data, const TensorShape<other_ndim> &shape) = delete;
  template <typename T, int other_ndim>
  DynamicTensorView(const T *data, const TensorShape<other_ndim> &shape,
                    DALIDataType type_id) = delete;
  template <typename T>
  explicit DynamicTensorView(const TensorView<Backend, const T, ndim> &other) = delete;
  template <typename T, int other_ndim>
  explicit DynamicTensorView(const TensorView<Backend, const T, other_ndim> &other) = delete;
  // @}

  /**
   * @name Convert to strongly-typed TensorView
   *
   * Requires match between runtime and static types, no-op if DataType = DynamicType
   */
  // @{
  template <typename DataType, int other_ndim = ndim>
  std::enable_if_t<!std::is_same<std::remove_const_t<DataType>, DynamicType>::value,
                   TensorView<Backend, DataType, other_ndim>>
  to_static_type() const & {
    DALI_ENFORCE(type_id == TypeTable::GetTypeId<DataType>(),
                 make_string("Calling type does not match view data type, requested type: ",
                             TypeTable::GetTypeId<DataType>(), " current view type: ", type_id));
    detail::check_compatible_ndim<ndim, other_ndim>();
    return {static_cast<DataType *>(data), shape};
  }

  template <typename DataType, int other_ndim = ndim>
  std::enable_if_t<!std::is_same<std::remove_const_t<DataType>, DynamicType>::value,
                   TensorView<Backend, DataType, other_ndim>>
  to_static_type() && {
    DALI_ENFORCE(type_id == TypeTable::GetTypeId<DataType>(),
                 make_string("Calling type does not match view data type, requested type: ",
                             TypeTable::GetTypeId<DataType>(), " current view type: ", type_id));
    detail::check_compatible_ndim<ndim, other_ndim>();
    return {static_cast<DataType *>(data), std::move(shape)};
  }

  template <typename DataType, int other_ndim = ndim>
  std::enable_if_t<std::is_same<std::remove_const_t<DataType>, DynamicType>::value,
                   TensorView<Backend, DataType, other_ndim>>
  to_static_type() const & {
    detail::check_compatible_ndim<ndim, other_ndim>();
    return {data, shape, type_id};
  }

  template <typename DataType, int other_ndim = ndim>
  std::enable_if_t<std::is_same<std::remove_const_t<DataType>, DynamicType>::value,
                   TensorView<Backend, DataType, other_ndim>>
  to_static_type() && {
    detail::check_compatible_ndim<ndim, other_ndim>();
    return {data, std::move(shape), type_id};
  }
  // @}

  /**
   * @brief Change the ndim, compatible with TensorView API
   */
  template <int other_ndim>
  DynamicTensorView<Backend, other_ndim> to_static() const {
    static_assert(other_ndim != DynamicDimensions,
                  "Conversion to static only allowed for static shape");
    static_assert(ndim == other_ndim || ndim == DynamicDimensions, "Cannot convert to other ndim");
    return {data, shape.template to_static<other_ndim>(), type_id};
  }

  using Base::data;
  using Base::shape;
  using Base::type_id;
};

template <typename Backend, int ndim = DynamicDimensions>
struct ConstDynamicTensorView : DynamicTensorViewBase<Backend, const void, ndim> {
  using Base = DynamicTensorViewBase<Backend, const void, ndim>;

  ConstDynamicTensorView() = default;

  /**
   * @name Construct the view inferring the type_id from the pointer value.
   */
  // @{
  template <typename T>
  ConstDynamicTensorView(T *data, const TensorShape<ndim> &shape)
      : Base(data, shape, TypeTable::GetTypeId<std::remove_const_t<T>>()) {}

  template <typename T>
  ConstDynamicTensorView(T *data, TensorShape<ndim> &&shape)
      : Base(data, std::move(shape), TypeTable::GetTypeId<std::remove_const_t<T>>()) {}

  template <typename T, int other_ndim>
  ConstDynamicTensorView(T *data, const TensorShape<other_ndim> &shape)
      : Base(data, shape, TypeTable::GetTypeId<std::remove_const_t<T>>()) {
    detail::check_compatible_ndim<ndim, other_ndim>();
  }

  template <typename T, int other_ndim>
  ConstDynamicTensorView(T *data, TensorShape<other_ndim> &&shape)
      : Base(data, std::move(shape), TypeTable::GetTypeId<std::remove_const_t<T>>()) {
    detail::check_compatible_ndim<ndim, other_ndim>();
  }
  // @}


  /**
   * @name Construct the view with explicitly provided type_id.
   */
  // @{
  ConstDynamicTensorView(const void *data, const TensorShape<ndim> &shape, DALIDataType type_id)
      : Base(data, shape, type_id) {}

  ConstDynamicTensorView(const void *data, TensorShape<ndim> &&shape, DALIDataType type_id)
      : Base(data, std::move(shape), type_id) {}

  template <int other_ndim>
  ConstDynamicTensorView(const void *data, const TensorShape<other_ndim> &shape,
                         DALIDataType type_id)
      : Base(data, shape, type_id) {
    detail::check_compatible_ndim<ndim, other_ndim>();
  }

  template <int other_ndim>
  ConstDynamicTensorView(const void *data, TensorShape<other_ndim> &&shape, DALIDataType type_id)
      : Base(data, std::move(shape), type_id) {
    detail::check_compatible_ndim<ndim, other_ndim>();
  }
  // @}

  /**
   * @name nullptr overloads with DALI_NO_TYPE
   */
  // @{
  ConstDynamicTensorView(std::nullptr_t *, const TensorShape<ndim> &shape)
      : Base(nullptr, shape, DALI_NO_TYPE) {}

  ConstDynamicTensorView(std::nullptr_t *, TensorShape<ndim> &&shape)
      : Base(nullptr, std::move(shape), DALI_NO_TYPE) {}

  template <int other_ndim>
  ConstDynamicTensorView(std::nullptr_t *, const TensorShape<other_ndim> &shape)
      : Base(nullptr, shape, DALI_NO_TYPE) {
    detail::check_compatible_ndim<ndim, other_ndim>();
  }

  template <int other_ndim>
  ConstDynamicTensorView(std::nullptr_t *, TensorShape<other_ndim> &&shape)
      : Base(nullptr, std::move(shape), DALI_NO_TYPE) {
    detail::check_compatible_ndim<ndim, other_ndim>();
  }
  // @}

  /**
   * @name Copy and move constructor and assignment ops.
   */
  // @{
  template <typename T, int other_ndim>
  explicit ConstDynamicTensorView(const DynamicTensorViewBase<Backend, T, other_ndim> &other) {
    detail::check_compatible_ndim<ndim, other_ndim>();
    this->data = other.data;
    this->shape = other.shape;
    this->type_id = other.type_id;
  }

  template <typename T, int other_ndim>
  ConstDynamicTensorView &operator=(const DynamicTensorViewBase<Backend, T, other_ndim> &other) {
    detail::check_compatible_ndim<ndim, other_ndim>();
    this->data = other.data;
    this->shape = other.shape;
    this->type_id = other.type_id;
    return *this;
  }

  template <typename T, int other_ndim>
  explicit ConstDynamicTensorView(DynamicTensorViewBase<Backend, T, other_ndim> &&other) {
    detail::check_compatible_ndim<ndim, other_ndim>();
    this->data = other.data;
    other.data = nullptr;
    this->shape = std::move(other.shape);
    this->type_id = other.type_id;
    other.type_id = DALI_NO_TYPE;
  }

  template <typename T, int other_ndim>
  ConstDynamicTensorView &operator=(DynamicTensorViewBase<Backend, T, other_ndim> &&other) {
    detail::check_compatible_ndim<ndim, other_ndim>();
    this->data = other.data;
    other.data = nullptr;
    this->shape = std::move(other.shape);
    this->type_id = other.type_id;
    other.type_id = DALI_NO_TYPE;
    return *this;
  }
  // @}


  /**
   * @name Converters from static TensorView
   */
  // @{
  template <typename T, int other_ndim,
            typename = std::enable_if_t<!std::is_same<std::remove_const_t<T>, DynamicType>::value>>
  explicit ConstDynamicTensorView(const TensorView<Backend, T, other_ndim> &other) {
    detail::check_compatible_ndim<ndim, other_ndim>();
    this->data = other.data;
    this->shape = other.shape;
    this->type_id = TypeTable::GetTypeId<remove_const_t<T>>();
  }

  template <typename T, int other_ndim,
            typename = std::enable_if_t<!std::is_same<std::remove_const_t<T>, DynamicType>::value>>
  explicit ConstDynamicTensorView(TensorView<Backend, T, other_ndim> &&other) {
    detail::check_compatible_ndim<ndim, other_ndim>();
    this->data = other.data;
    other.data = nullptr;
    this->shape = std::move(other.shape);
    this->type_id = TypeTable::GetTypeId<remove_const_t<T>>();
  }

  template <typename T, int other_ndim,
            typename = std::enable_if_t<!std::is_same<std::remove_const_t<T>, DynamicType>::value>>
  ConstDynamicTensorView &operator=(const TensorView<Backend, T, other_ndim> &other) {
    detail::check_compatible_ndim<ndim, other_ndim>();
    this->data = other.data;
    this->shape = other.shape;
    this->type_id = TypeTable::GetTypeId<remove_const_t<T>>();
    return *this;
  }

  template <typename T, int other_ndim,
            typename = std::enable_if_t<!std::is_same<std::remove_const_t<T>, DynamicType>::value>>
  ConstDynamicTensorView &operator=(TensorView<Backend, T, other_ndim> &&other) {
    detail::check_compatible_ndim<ndim, other_ndim>();
    this->data = other.data;
    other.data = nullptr;
    this->shape = std::move(other.shape);
    this->type_id = TypeTable::GetTypeId<remove_const_t<T>>();
    return *this;
  }
  // @}

  /**
   * @name Convert to strongly-typed TensorView
   *
   * Requires match between runtime and static types, no-op if DataType = DynamicType
   */
  // @{
  template <typename DataType, int other_ndim = ndim>
  std::enable_if_t<!std::is_same<std::remove_const_t<DataType>, DynamicType>::value,
                   TensorView<Backend, DataType, other_ndim>>
  to_static_type() const & {
    DALI_ENFORCE(type_id == TypeTable::GetTypeId<DataType>(),
                 make_string("Calling type does not match view data type, requested type: ",
                             TypeTable::GetTypeId<DataType>(), " current view type: ", type_id));
    static_assert(std::is_const<DataType>::value,
                  "This view contains a pointer to const, so the target type must also be const.");
    detail::check_compatible_ndim<ndim, other_ndim>();
    return {static_cast<DataType *>(data), shape};
  }

  template <typename DataType, int other_ndim = ndim>
  std::enable_if_t<!std::is_same<std::remove_const_t<DataType>, DynamicType>::value,
                   TensorView<Backend, DataType, other_ndim>>
  to_static_type() && {
    DALI_ENFORCE(type_id == TypeTable::GetTypeId<DataType>(),
                 make_string("Calling type does not match view data type, requested type: ",
                             TypeTable::GetTypeId<DataType>(), " current view type: ", type_id));
    static_assert(std::is_const<DataType>::value,
                  "This view contains a pointer to const, so the target type must also be const.");
    detail::check_compatible_ndim<ndim, other_ndim>();
    return {static_cast<DataType *>(data), std::move(shape)};
  }

  template <typename DataType, int other_ndim = ndim>
  std::enable_if_t<std::is_same<std::remove_const_t<DataType>, DynamicType>::value,
                   TensorView<Backend, DataType, other_ndim>>
  to_static_type() const & {
    static_assert(std::is_const<DataType>::value,
                  "This view contains a pointer to const, so the target type must also be const.");
    detail::check_compatible_ndim<ndim, other_ndim>();
    return {data, shape, type_id};
  }

  template <typename DataType, int other_ndim = ndim>
  std::enable_if_t<std::is_same<std::remove_const_t<DataType>, DynamicType>::value,
                   TensorView<Backend, DataType, other_ndim>>
  to_static_type() && {
    static_assert(std::is_const<DataType>::value,
                  "This view contains a pointer to const, so the target type must also be const.");
    detail::check_compatible_ndim<ndim, other_ndim>();
    return {data, std::move(shape), type_id};
  }
  // @}

  /**
   * @brief Change the ndim, compatible with TensorView API
   */
  template <int other_ndim>
  ConstDynamicTensorView<Backend, other_ndim> to_static() const {
    static_assert(other_ndim != DynamicDimensions,
                  "Conversion to static only allowed for static shape");
    static_assert(ndim == other_ndim || ndim == DynamicDimensions, "Cannot convert to other ndim");
    return {data, shape.template to_static<other_ndim>(), type_id};
  }

  using Base::data;
  using Base::shape;
  using Base::type_id;
};


/**
 * @brief TensorView keeping the type information as dynamic `type_id` field.
 */
template <typename Backend, int ndim>
struct TensorView<Backend, DynamicType, ndim> : public DynamicTensorView<Backend, ndim> {
  using DynamicTensorView<Backend, ndim>::DynamicTensorView;
};


/**
 * @brief TensorView keeping the type information as dynamic `type_id` field, with pointer to const.
 */
template <typename Backend, int ndim>
struct TensorView<Backend, const DynamicType, ndim> : public ConstDynamicTensorView<Backend, ndim> {
  using ConstDynamicTensorView<Backend, ndim>::ConstDynamicTensorView;
};


/**
 * @brief TensorView keeping the type information as dynamic `type_id` field.
 *
 * Default variant with dynamic shape (DynamicDimensions specialization).
 */
template <typename Backend>
struct TensorView<Backend, DynamicType, DynamicDimensions>
    : public DynamicTensorView<Backend, DynamicDimensions> {
  using DynamicTensorView<Backend, DynamicDimensions>::DynamicTensorView;
};


/**
 * @brief TensorView keeping the type information as dynamic `type_id` field, with pointer to const.
 *
 * Default variant with dynamic shape (DynamicDimensions specialization).
 */
template <typename Backend>
struct TensorView<Backend, const DynamicType, DynamicDimensions>
    : public ConstDynamicTensorView<Backend, DynamicDimensions> {
  using ConstDynamicTensorView<Backend, DynamicDimensions>::ConstDynamicTensorView;
};


}  // namespace dali

#endif  // DALI_PIPELINE_DATA_DYNAMIC_TENSOR_VIEW_H_
