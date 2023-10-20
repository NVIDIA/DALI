// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_DATA_TENSOR_H_
#define DALI_PIPELINE_DATA_TENSOR_H_

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/util.h"
#include "dali/core/span.h"
#include "dali/core/traits.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/meta.h"

namespace dali {

/**
 * @brief Stores dense, multi-dimensional data. Provides utilities
 * methods for handling dimensions and shapes of the stored data.
 */
template <typename Backend>
class Tensor : public Buffer<Backend> {
 public:
  inline Tensor() {}
  inline ~Tensor() override = default;


  /**
   *
   * @brief For tensor T of shape (s_0, s_1, ..., s_{n-1}) returns a n-1 dimensional tensor T'
   *        of shape (s_1, s_2, ..., s_{n-1}), such that
   *        T'(x_1, x_2, ..., x_{n-1}) = T(x, x_1, x_2, ..., x_{n-1})
   *        for param 'x' and any valid x_1, x_2, ..., x_{n-1}
   *
   * Tensor should have at least 2 dimensions.
   * Returned tensor is treated as a view to this tensors and shares memory with it.
   * @param x Subspace between 0 and dim(0) - 1.
   * @return Tensor<Backend>
   */
  Tensor<Backend> SubspaceTensor(Index x) {
    DALI_ENFORCE(ndim() > 1,
                 "To obtain subspace tensor, source tensor should have at least 2 dimensions");
    DALI_ENFORCE(0 <= x && x < dim(0), "'x' should be valid index to first dimension: [0, dim(0))");
    Tensor<Backend> view;
    view.shape_ = shape_.last(shape_.size() - 1);
    view.type_ = type_;
    view.size_ = size_ / shape_[0];
    view.num_bytes_ = view.type_.size() * view.size_;
    // Point to the data using proper aliasing shared_ptr
    auto *data_ptr = static_cast<uint8_t *>(this->raw_mutable_data()) + x * view.num_bytes_;
    view.data_ = std::shared_ptr<void>(data_, data_ptr);
    view.shares_data_ = true;
    view.device_ = device_;
    return view;
  }

  /**
   * Loads the Tensor with data from the input vector.
   */
  template <typename T>
  inline void Copy(const vector<T> &data, AccessOrder order = {}) {
    this->Resize({(Index)data.size()}, TypeTable::GetTypeId<T>());
    if (!order)
      order = std::is_same<Backend, CPUBackend>::value ? AccessOrder::host() : order_;

    order.wait(order_);

    type_.template Copy<Backend, CPUBackend>(this->raw_mutable_data(),
        data.data(), this->size(), order.stream());
    order_.wait(order);
  }

  /**
   * Loads the Tensor with data from a span.
   */
  template <typename T>
  inline void Copy(span<T> data, AccessOrder order = {}) {
    using U = remove_const_t<T>;
    this->Resize({(Index)data.size()}, TypeTable::GetTypeId<U>());
    if (!order)
      order = std::is_same<Backend, CPUBackend>::value ? AccessOrder::host() : order_;

    order.wait(order_);
    type_.template Copy<Backend, CPUBackend>(this->raw_mutable_data(),
        data.data(), this->size(), order.stream());
    order_.wait(order);
  }

  /**
   * Loads the Tensor with data from the input Tensor.
   */
  template <typename InBackend>
  inline void Copy(const Tensor<InBackend> &other, AccessOrder order = {}) {
    constexpr bool is_host_to_host = std::is_same<Backend, CPUBackend>::value &&
                                     std::is_same<InBackend, CPUBackend>::value;
    if (!order) {
      if (is_host_to_host)
        order = AccessOrder::host();
      else
        order = other.order() ? other.order() : order_;
    }
    DALI_ENFORCE(!is_host_to_host || !order.is_device(),
                 "Cannot issue a host-to-host copy on a device stream.");
    this->Resize(other.shape(), other.type());
    order.wait(order_);
    this->SetLayout(other.GetLayout());
    this->SetSourceInfo(other.GetSourceInfo());
    this->SetSkipSample(other.ShouldSkipSample());
    type_.template Copy<Backend, InBackend>(this->raw_mutable_data(),
        other.raw_data(), this->size(), order.stream());
    order_.wait(order);
  }

  /**
   * @brief Resizes the buffer to fit `volume(shape)` elements.
   * The underlying storage is only reallocated in the case that
   * the current buffer is not large enough for the requested
   * number of elements.
   */
  inline void Resize(const TensorShape<> &shape) {
    DALI_ENFORCE(IsValidType(type_),
                 "Tensor has no type, 'set_type<T>()' or Resize(shape, type) must be called "
                 "on the Tensor to set a valid type before it can be resized.");
    Index new_size = volume(shape);
    resize(new_size);
    shape_ = shape;
  }

  /**
   * @brief Resizes the buffer to fit `volume(shape)` of new_type elements.
   * The underlying storage is only reallocated in the case that
   * the current buffer is not large enough for the requested
   * number of elements.
   */
  inline void Resize(const TensorShape<> &shape, DALIDataType new_type) {
    DALI_ENFORCE(IsValidType(new_type),
                 "Tensor cannot be resized with invalid type. To zero out the Tensor "
                 "Reset() can be used.");
    Index new_size = volume(shape);
    resize(new_size, new_type);
    shape_ = shape;
  }

  /**
   * @brief Tensor is always backed by contiguous buffer
   */
  bool IsContiguous() const {
    return true;
  }

  /**
   * @brief Tensor is always backed by contiguous buffer
   *        Cannot be set to noncontiguous
   */
  void SetContiguity(BatchContiguity state) {
    DALI_ENFORCE(state != BatchContiguity::Noncontiguous, "Tensor cannot be made noncontiguous");
  }

  using Buffer<Backend>::reserve;

  // For having complete API, Tensor is not a batch
  void reserve(size_t bytes_per_tensor, int) {
    reserve(bytes_per_tensor);
  }

  /**
   * @brief Wraps the data owned by the input tensor. The input
   * tensor must have a valid type. If successful, the tensor
   * object will wrap the target data and assume the datatype
   * and shape of the data stored in the Tensor.
   * Additionally, the tensor will assume target's order (stream or host).
   *
   * If the input does not store any data, shares_data_ is left
   * as false.
   *
   * After calling this function any following call to `set_type` and `Resize`
   * must match the total size of underlying allocation (`num_bytes_`) of
   * shared data or the call will fail.
   */
  inline void ShareData(const Tensor<Backend> &t) {
    if (this == &t)
      return;

    DALI_ENFORCE(IsValidType(t.type()), "To share data, "
        "the input Tensor must have a valid data type.");

    Buffer<Backend>::ShareData(t);

    // Copy the tensor's meta-data
    shape_ = t.shape_;
    meta_ = t.meta_;
  }

  /**
   * @brief Interprets a raw allocation as a tensor with given shape.
   *
   * If the size of the allocation is zero, the Tensor is reset to a default
   * state and is NOT marked as sharing data.
   *
   * After calling this function any following call to `set_type` and `Resize`
   * must not exceed the total size of underlying allocation (`num_bytes_`) of
   * shared data or the call will fail.
   * Size can be set to 0 and type to NoType as intermediate step.
   *
   * @remark Note that the device_id inside the order can differ from the device_id that is passed
   * individually. The device_id describes the location of the memory and the order can describe
   * the dependency on the work that is happening on another device.
   */
  inline void ShareData(const shared_ptr<void> &ptr, size_t bytes, bool pinned,
                        const TensorShape<> &shape, DALIDataType type, int device_id,
                        AccessOrder order = {}) {
    Index new_size = volume(shape);
    DALI_ENFORCE(new_size == 0 || type != DALI_NO_TYPE,
      "Only empty tensors can be shared without specifying a type.");

    // Free the underlying storage.
    if (!same_managed_object(data_, ptr))
      free_storage();

    // Set the new order, if provided.
    if (order)
      this->set_order(order);

    // Save our new pointer and bytes. Reset our type, shape, and size
    type_ = TypeTable::GetTypeInfo(type);
    data_ = ptr;
    size_ = new_size;
    num_bytes_ = bytes;
    device_ = device_id;

    // If the input pointer stores a non-zero size allocation, mark
    // that we are sharing our underlying data
    shares_data_ = num_bytes_ > 0 ? true : false;
    pinned_ = pinned;

    shape_ = shape;
  }

  /**
   * @brief Interprets a raw allocation as a tensor with given shape.
   *
   * If the size of the allocation is zero, the Tensor is reset to a default
   * state and is NOT marked as sharing data.
   *
   * After calling this function any following call to `set_type` and `Resize`
   * must not exceed the total size of underlying allocation (`num_bytes_`) of
   * shared data or the call will fail.
   * Size can be set to 0 and type to NoType as intermediate step.
   *
   * The Tensor object assumes no ownership of the input allocation, and will
   * not de-allocate it when it is done using it. It is up to the user to
   * manage the lifetime of the allocation such that it persist while it is
   * in use by the Tensor.
   *
   * @remark Note that the device_id inside the order can differ from the device_id that is passed
   * individually. The device_id describes the location of the memory and the order can describe
   * the dependency on the work that is happening on another device.
   */
  inline void ShareData(void *ptr, size_t bytes, bool pinned, const TensorShape<> &shape,
                        DALIDataType type, int device_id, AccessOrder order = {}) {
    ShareData(shared_ptr<void>(ptr, [](void *) {}), bytes, pinned, shape, type, device_id, order);
  }

  /**
   * @brief Wraps the raw allocation. The input pointer must not be nullptr.
   * if the size of the allocation is zero, the Tensor is reset to a default
   * state and is NOT marked as sharing data.
   *
   * After wrapping the allocation, the Tensors size is set to 0, and its
   * type is reset to NoType.
   * After calling this function any following call to `set_type` and `Resize`
   * must match the total size of underlying allocation (`num_bytes_`) of
   * shared data or the call will fail.
   * Size can be set to 0 and type to NoType as intermediate step.
   *
   * The Tensor object assumes no ownership of the input allocation, and will
   * not de-allocate it when it is done using it. It is up to the user to
   * manage the lifetime of the allocation such that it persist while it is
   * in use by the Tensor.
   *
   * @remark Note that the device_id inside the order can differ from the device_id that is passed
   * individually. The device_id describes the location of the memory and the order can describe
   * the dependency on the work that is happening on another device.
   */
  inline void ShareData(void *ptr, size_t bytes, bool pinned, DALIDataType type, int device_id,
                        AccessOrder order = {}) {
    ShareData(ptr, bytes, pinned, { 0 }, type, device_id, order);
  }

  inline void Reset(AccessOrder order = {}) {
    reset(order);  // free the underlying buffer
    shape_ = { 0 };
    meta_ = {};
  }

  /**
   * @brief Returns the shape of the Tensor
   */
  inline const TensorShape<> &shape() const {
    return shape_;
  }

  /**
   * @brief Returns the number of dimensions of the Tensor
   */
  inline virtual int ndim() const {
    return shape_.size();
  }

  /**
   * @brief Returns the size of the dimension at the given index
   */
  inline virtual Index dim(int idx) const {
#ifndef NDEBUG
    DALI_ENFORCE(idx < shape_.size(), "index exceeds ndim");
    DALI_ENFORCE(idx >= 0, "negative index not supported");
#endif
    return shape_[idx];
  }

  /**
   * @brief Remove any single-dimensional entries from the shape
   * of a Tensor.
   * @returns true if the shape changed, false otherwise.
   */
  inline bool Squeeze() {
    DynamicTensorShapeContainer out_shape;
    TensorLayout out_layout;
    TensorLayout in_layout = GetLayout();
    bool is_squeezed = false;
    for (int d = 0; d < shape_.size(); d++) {
      if (shape_[d] == 1) {
        is_squeezed = true;
        continue;
      }
      out_shape.push_back(shape_[d]);
      if (!in_layout.empty())
        out_layout += in_layout[d];
    }
    shape_ = std::move(out_shape);
    SetLayout(out_layout);
    return is_squeezed;
  }

  /**
   * @brief Removes the specified dimension from the shape, if its extent is
   * equal to 1.
   * @param dim Dimension to be squeezed. Negative indexing is also supported
   * @returns true if the shape changed, false otherwise.
   */
  inline bool Squeeze(int dim) {
    int ndim = shape_.size();
    DALI_ENFORCE(dim >= -ndim && dim <= (ndim - 1),
                 make_string("axis ", dim, " is out of bound for a tensor with ", shape_.size(),
                             " dimensions."));
    if (dim < 0) {
      dim += shape_.size();
    }
    if (shape_[dim] == 1) {
      shape_.shape.erase(shape_.shape.begin() + dim);
      auto layout = GetLayout();
      if (!layout.empty()) {
        layout.erase(dim);
        SetLayout(layout);
      }
      return true;
    }
    return false;
  }

  /**
   * @brief Compares the shape of this tensor against another tensor,
   * returning equality
   */
  template <typename OtherBackend>
  inline bool SameShape(const Tensor<OtherBackend> &other) const {
    if (this->ndim() != other.ndim()) return false;

    for (int i = 0; i < ndim(); ++i) {
      if (this->dim(i) != other.dim(i)) return false;
    }
    return true;
  }

  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  Tensor(Tensor &&t) noexcept {
    *this = std::move(t);
  }

  Tensor& operator=(Tensor &&t) noexcept {
    if (&t != this) {
      shape_ = std::exchange(t.shape_, {0});
      meta_ = std::exchange(t.meta_, {});
      move_buffer(std::move(t));
    }
    return *this;
  }

  const DALIMeta &GetMeta() const {
    return meta_;
  }

  void SetMeta(const DALIMeta &meta)  {
    meta_ = meta;
  }

  inline TensorLayout GetLayout() const {
    return meta_.GetLayout();
  }

  inline void SetLayout(const TensorLayout &layout) {
    meta_.SetLayout(layout);
  }

  inline string GetSourceInfo() const {
    return meta_.GetSourceInfo();
  }

  inline void SetSourceInfo(const string &source_info) {
    meta_.SetSourceInfo(source_info);
  }

  inline void SetSkipSample(bool skip_sample) {
    meta_.SetSkipSample(skip_sample);
  }

  inline bool ShouldSkipSample() const {
    return meta_.ShouldSkipSample();
  }

 protected:
  TensorShape<> shape_ = { 0 };
  DALIMeta meta_;
  USE_BUFFER_MEMBERS();

  // So TensorList can access data_ of the tensor directly
  template <typename InBackend>
  friend class TensorList;
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_H_
