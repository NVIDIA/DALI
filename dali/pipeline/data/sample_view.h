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

#ifndef DALI_PIPELINE_DATA_SAMPLE_VIEW_H_
#define DALI_PIPELINE_DATA_SAMPLE_VIEW_H_

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/types.h"

namespace dali {

/**
 * @brief SampleView - non-owning view type, keeping the pointer, shape and runtime type information
 * of specific sample.
 *
 * This is a type corresponding to Tensor<Backend> and it's main goal is to become the return type
 * for the TensorList/TensorList::operator[]. It allows to access the data via the
 * convenient `view<T, ndim>(SampleView)` conversion to TensorView, but doesn't break the batch
 * object encapsulation and doesn't allow to adjust the allocation.
 */
template <typename Backend, typename ptr_t>
class SampleViewBase {
 public:
  /**
   * @name Get the underlying pointer to data
   * @{
   */
  /**
   * @brief Return an un-typed pointer to the underlying storage.
   */
  template <typename ptr_t_ = ptr_t>
  std::enable_if_t<std::is_same<ptr_t_, void *>::value, void *> raw_mutable_data() {
    return data_;
  }

  /**
   * @brief Return a const, un-typed pointer to the underlying storage.
   */
  const void *raw_data() const {
    return data_;
  }

  /**
   * @brief Returns a typed pointer to the underlying storage.
   * The calling type must match the underlying type of the buffer.
   */
  template <typename T, typename ptr_t_ = ptr_t>
  inline std::enable_if_t<std::is_same<ptr_t_, void *>::value, T *> mutable_data() {
    DALI_ENFORCE(
        type() == TypeTable::GetTypeId<T>(),
        make_string(
            "Calling type does not match buffer data type, requested type: ",
            TypeTable::GetTypeId<T>(), " current buffer type: ", type(),
            ". To set type for the Buffer use 'set_type<T>()' or Resize(shape, type) first."));
    return static_cast<T *>(data_);
  }

  /**
   * @brief Returns a const, typed pointer to the underlying storage.
   * The calling type must match the underlying type of the buffer.
   */
  template <typename T>
  inline const T *data() const {
    DALI_ENFORCE(
        type() == TypeTable::GetTypeId<T>(),
        make_string(
            "Calling type does not match buffer data type, requested type: ",
            TypeTable::GetTypeId<T>(), " current buffer type: ", type(),
            ". To set type for the Buffer use 'set_type<T>()' or Resize(shape, type) first."));
    return static_cast<const T *>(data_);
  }
  /** @} */

  /**
   * @brief Get the shape of the sample
   */
  const TensorShape<> &shape() const {
    return shape_;
  }

  /**
   * @brief Get the runtime information about the type of the sample
   */
  DALIDataType type() const {
    return type_id_;
  }


  SampleViewBase() = default;

  SampleViewBase(const SampleViewBase &) = default;
  SampleViewBase &operator=(const SampleViewBase &) = default;

  SampleViewBase(SampleViewBase &&other) {
    *this = std::move(other);
  }

  SampleViewBase &operator=(SampleViewBase &&other) {
    if (this != &other) {
      data_ = std::exchange(other.data_, nullptr);
      shape_ = std::exchange(other.shape_, {0});
      type_id_ = std::exchange(other.type_id_, DALI_NO_TYPE);
    }
    return *this;
  }

  /**
   * @brief Construct the view inferring the type_id from the pointer value.
   */
  template <typename T>
  SampleViewBase(T *data, TensorShape<> shape)
      : data_(data),
        shape_(std::move(shape)),
        type_id_(TypeTable::GetTypeId<std::remove_const_t<T>>()) {}

  /**
   * @brief Construct the view with explicitly provided type_id.
   */
  SampleViewBase(ptr_t data, const TensorShape<> shape, DALIDataType type_id)
      : data_(data), shape_(std::move(shape)), type_id_(type_id) {}

 protected:
  // TODO(klecki): The view is introduced with no co-owning pointer, it will be evaluated
  // if the usage of shared_ptr is possbile and adjusted if necessary.
  // Using shared_ptr might allow for sample exchange between two batches using operator[]
  ptr_t data_ = nullptr;
  TensorShape<> shape_ = {0};
  DALIDataType type_id_ = DALI_NO_TYPE;
  ~SampleViewBase() = default;
};

template <typename Backend>
class ConstSampleView;

template <typename Backend>
class SampleView : public SampleViewBase<Backend, void *> {
 public:
  using Base = SampleViewBase<Backend, void *>;
  using Base::Base;

 private:
  using Base::data_;
  using Base::shape_;
  using Base::type_id_;
  friend class ConstSampleView<Backend>;
};


template <typename Backend>
class ConstSampleView : public SampleViewBase<Backend, const void *> {
 public:
  using Base = SampleViewBase<Backend, const void *>;
  using Base::Base;

  ConstSampleView(const SampleView<Backend> &other)  // NOLINT
      : Base(other.raw_data(), other.shape(), other.type()) {}

  ConstSampleView &operator=(const SampleView<Backend> &other) {
    data_ = other.raw_data();
    shape_ = other.shape();
    type_id_ = other.type();
    return *this;
  }

  ConstSampleView(SampleView<Backend> &&other) {  // NOLINT
    *this = std::move(other);
  }

  ConstSampleView &operator=(SampleView<Backend> &&other) {
    data_ = std::exchange(other.data_, nullptr);
    shape_ = std::exchange(other.shape_, {0});
    type_id_ = std::exchange(other.type_id_, DALI_NO_TYPE);
    return *this;
  }

 private:
  using Base::data_;
  using Base::shape_;
  using Base::type_id_;
};


}  // namespace dali

#endif  // DALI_PIPELINE_DATA_SAMPLE_VIEW_H_
