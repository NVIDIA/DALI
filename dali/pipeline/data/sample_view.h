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
 * for the TensorVector/TensorList::operator[]. It allows to access the data via the
 * convenient `view<T, ndim>(SampleView)` conversion to TensorView, but doesn't break the batch
 * object encapsulation and doesn't allow to adjust the allocation.
 */
template <typename Backend>
class SampleView {
 public:
  /**
   * @name Get the underlying pointer to data
   */
  // @{
  /**
   * @brief Return an un-typed pointer to the underlying storage.
   */
  void *raw_mutable_data() {
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
  template <typename T>
  inline T *mutable_data() {
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
    return static_cast<T *>(data_);
  }
  //@}

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


  SampleView() = default;

  SampleView(const SampleView &) = default;
  SampleView &operator=(const SampleView &) = default;

  SampleView(SampleView &&other) {
    *this = std::move(other);
  }

  SampleView &operator=(SampleView &&other) {
    if (this != &other) {
      data_ = other.data_;
      other.data_ = nullptr;
      shape_ = std::move(other.shape_);
      other.shape_ = {0};
      type_id_ = other.type_id_;
      other.type_id_ = DALI_NO_TYPE;
    }
    return *this;
  }

  /**
   * @brief Construct the view inferring the type_id from the pointer value.
   */
  template <typename T>
  SampleView(T *data, TensorShape<> shape)
      : data_(data), shape_(std::move(shape)), type_id_(TypeTable::GetTypeId<T>()) {}

  /**
   * @brief Construct the view with explicitly provided type_id.
   */
  SampleView(void *data, const TensorShape<> shape, DALIDataType type_id)
      : data_(data), shape_(std::move(shape)), type_id_(type_id) {}

 private:
  // TODO(klecki): The view is introduced with no co-owning pointer, it will be evaluated
  // if the usage of shared_ptr is possbile and adjusted if necessary.
  // Using shared_ptr might allow for sample exchange between two batches using operator[]
  void *data_ = nullptr;
  TensorShape<> shape_ = {0};
  DALIDataType type_id_ = DALI_NO_TYPE;
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_SAMPLE_VIEW_H_
