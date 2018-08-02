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

#ifndef DALI_PIPELINE_DATA_TENSOR_H_
#define DALI_PIPELINE_DATA_TENSOR_H_

#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/tensor_list.h"

namespace dali {

/**
 * @brief Stores dense, multi-dimensional data. Provides utilities
 * methods for handling dimensions and shapes of the stored data.
 */
template <typename Backend>
class Tensor {
 public:
  DLL_PUBLIC inline Tensor() : layout_(DALI_NHWC), pinned_(false) {
    buffer_.reset();
  }
  DLL_PUBLIC virtual inline ~Tensor() = default;

  /**
   * Loads the Tensor with data from the input vector.
   */
  template <typename T>
  DLL_PUBLIC inline void Copy(const vector<T> &data, cudaStream_t stream) {
    set_type(TypeInfo::Create<T>());
    Resize({(Index)data.size()});
    acquire_buffer();
    buffer_->type().template Copy<Backend, CPUBackend>(raw_mutable_data(),
        data.data(), buffer_->size(), stream);
  }

  /**
   * Loads the Tensor with data from the input Tensor.
   */
  template <typename InBackend>
  DLL_PUBLIC inline void Copy(const Tensor<InBackend> &other, cudaStream_t stream) {
    set_type(other.type());
    ResizeLike(other);
    acquire_buffer();
    // buffer_->template mutable_data
    // buffer_->set_type(other.type());
    buffer_->type().template Copy<Backend, InBackend>(raw_mutable_data(),
        other.raw_data(), buffer_->size(), stream);
  }

  /**
   * @brief Loads the Tensor at index idx from the input TensorList.
   */
  template <typename InBackend>
  inline void Copy(const TensorList<InBackend> &other, int idx, cudaStream_t stream) {
    shape_ = other.tensor_shape(idx);
    device_ = other.device_id();
    this->set_type(other.type());
    this->Resize(shape_);
    type_.template Copy<Backend, InBackend>(this->raw_mutable_data(),
        other.raw_tensor(idx), this->size(), stream);
  }

  template <typename InBackend>
  DLL_PUBLIC inline void ResizeLike(const Tensor<InBackend> &other) {
    Resize(other.shape());
  }

  /**
   * @brief Resizes the buffer to fit `Product(shape)` elements.
   * The underlying storage is only reallocated in the case that
   * the current buffer is not large enough for the requested
   * number of elements.
   */
  DLL_PUBLIC void Resize(const vector<Index> &shape);

  /**
   * @brief Wraps the data owned by the tensor at the given index
   * in the input tensor list. The input tensor list must have
   * a valid type, and the given index must be in the valid range
   * [0, tl.ntensor()).
   *
   * If sucessful, the tensor object will wrap the target data and
   * assume the datatype of the data stored in the TensorList.
   *
   * Because we are storing the pointer of the TensorList at an
   * offset, we do not guarantee that this allocation will persist
   * until both the owner and the sharer are finished with it. Thus,
   * it is up to the user to manage the scope of the sharing objects
   * to ensure correctness.
   */
  DLL_PUBLIC inline void ShareData(TensorList<Backend> *tl, int idx) {
    buffer_.reset(new Buffer<Backend>);
    shares_data_ = true;

    DALI_ENFORCE(tl != nullptr, "Input TensorList is nullptr");
    DALI_ENFORCE(IsValidType(tl->type()), "To share data, "
        "the input TensorList must have a valid data type.");
    DALI_ENFORCE(idx >= 0, "Negative tensor index not supported.");
    DALI_ENFORCE(idx < tl->ntensor(), "Index of " + std::to_string(idx) +
        " out of range for TensorList of size " + std::to_string(tl->ntensor()));

    // Get the meta-data for the target tensor
    shape_ = tl->tensor_shape(idx);
    set_type(tl->type());
    auto size = Product(shape_);
    auto num_bytes = tl->type().size() * size;
    buffer_->ShareData(tl->raw_mutable_tensor(idx), num_bytes, tl->type());
  }

  /**
   * @brief Wraps the data owned by the input tensor. The input
   * tensor must have a valid type. If sucessful, the tensor
   * object will wrap the target data and assume the datatype
   * and shape of the data stored in the Tensor.
   *
   * If the input does not store any data, shares_data_ is left
   * as false.
   */
  DLL_PUBLIC inline void ShareData(Tensor<Backend> *t) {
    buffer_.reset(new Buffer<Backend>);
    shares_data_ = true;

    DALI_ENFORCE(t != nullptr, "Input Tensor is nullptr");
    DALI_ENFORCE(IsValidType(t->type()), "To share data, "
        "the input Tensor must have a valid data type.");

    // Save the tensor meta-data
    shape_ = t->shape_;
    set_type(t->type());
    auto size = t->size();
    auto num_bytes = t->nbytes();

    buffer_->ShareData(t->raw_mutable_data(), num_bytes, type());
  }

  /**
   * @brief Wraps the raw allocation. The input pointer must not be nullptr.
   * if the size of the allocation is zero, the Tensor is reset to a default
   * state and is NOT marked as sharing data.
   *
   * After wrapping the allocation, the Tensors size is set to 0, and its
   * type is reset to NoType. Future calls to Resize or setting of the
   * Tensor type will evaluate whether or not the current allocation is
   * large enough to be used and proceed appropriately.
   *
   * The Tensor object assumes no ownership of the input allocation, and will
   * not de-allocate it when it is done using it. It is up to the user to
   * manage the lifetime of the allocation such that it persist while it is
   * in use by the Tensor.
   */
  DLL_PUBLIC inline void ShareData(void *ptr, size_t bytes) {
    buffer_.reset(new Buffer<Backend>);
    shares_data_ = true;

    DALI_ENFORCE(ptr != nullptr, "Input pointer must not be nullptr.");

    // Save our new pointer and bytes. Reset our type, shape, and size
    shape_.clear();
    buffer_->ShareData(ptr, bytes);
  }

  /**
   * @brief Wraps a TensorList
   * TensorList has to be a valid tensor
   * (there must be at least 1 tensor stored in TensorList,
   * all shapes should be identical,
   * all tensors need to be stored without
   * any offset between them)
   */
  DLL_PUBLIC inline void ShareData(TensorList<Backend> *tl) {
    buffer_.reset(new Buffer<Backend>);
    shares_data_ = true;

    DALI_ENFORCE(tl != nullptr, "Input TensorList is nullptr");
    DALI_ENFORCE(IsValidType(tl->type()), "To share data, "
        "the input TensorList must have a valid data type.");
    DALI_ENFORCE(tl->IsDenseTensor(),
      "All tensors in the input TensorList must have the same shape and be densely packed.");
    DALI_ENFORCE(tl->ntensor() > 0, "Input TensorList has 0 elements!");

    // Get the meta-data for the target tensor
    shape_ = tl->tensor_shape(0);
    shape_.insert(shape_.begin(), tl->ntensor());
    set_type(tl->type());
    auto size = Product(shape_);
    auto num_bytes = type().size() * size;
    buffer_->ShareData(tl->raw_mutable_tensor(0), num_bytes, type());
  }

  /**
   * @brief Returns the shape of the Tensor
   */
  DLL_PUBLIC inline vector<Index> shape() const {
    return shape_;
  }

  /**
   * @brief Returns the number of dimensions of the Tensor
   */
  DLL_PUBLIC inline virtual int ndim() const {
    return shape_.size();
  }

  /**
   * @brief Returns the size of the dimension at the given index
   */
  DLL_PUBLIC inline virtual Index dim(int idx) const {
#ifndef NDEBUG
    DALI_ENFORCE((size_t)idx < shape_.size(), "index exceeds ndim");
    DALI_ENFORCE(idx >= 0, "negative index not supported");
#endif
    return shape_[idx];
  }

  /**
   * @brief Remove single-dimensional entries from the shape
   * of a Tensor
   */
  DLL_PUBLIC inline void Squeeze() {
    shape_.erase(std::remove(shape_.begin(), shape_.end(), 1), shape_.end());
    if (shape_.empty()) {
      shape_.push_back(1);
    }
  }

  /**
   * @brief Compares the shape of this tensor against another tensor,
   * returning equality
   */
  template <typename OtherBackend>
  DLL_PUBLIC inline bool SameShape(const Tensor<OtherBackend> &other) const {
    if (this->ndim() != other.ndim()) return false;

    for (int i = 0; i < ndim(); ++i) {
      if (this->dim(i) != other.dim(i)) return false;
    }
    return true;
  }

  Tensor<Backend>(const Tensor<Backend>&) = delete;
  Tensor<Backend>& operator=(const Tensor<Backend>&) = delete;

  DLL_PUBLIC Tensor<Backend>(Tensor<Backend> &&t) noexcept {
    // Steal all data and set input to default state
    type_ = std::move(t.type_);
    shape_ = std::move(t.shape_);
    buffer_ = std::move(t.buffer_);

    t.shape_.clear();
  }

  DLL_PUBLIC Tensor<Backend>& operator=(Tensor<Backend> &&t) noexcept {
    if (&t != this) {
      shape_ = std::move(t.shape_);
      buffer_ = std::move(t.buffer_);
      type_ = std::move(t.type_);

      t.shape_.clear();
    }
    return *this;
  }

  inline DALITensorLayout GetLayout() const {
    return layout_;
  }

  inline void SetLayout(DALITensorLayout layout) {
    layout_ = layout;
  }

  template <typename T>
  DLL_PUBLIC const T *data() const {
    if (!buffer_.get()) return nullptr;
    return buffer_->template data<T>();
  }

  template <typename T>
  DLL_PUBLIC T *mutable_data() {
    set_type(TypeInfo::Create<T>());
    acquire_buffer();
    if (!buffer_.get()) return nullptr;
    return buffer_->template mutable_data<T>();
  }

  DLL_PUBLIC const void *raw_data() const {
    if (!buffer_.get()) return nullptr;
    return buffer_->raw_data();
  }

  DLL_PUBLIC void *raw_mutable_data() {
    acquire_buffer();
    if (!buffer_.get()) {
      return nullptr;
    }
    return buffer_->raw_mutable_data();
  }

  DLL_PUBLIC TypeInfo type() const {
    return type_;
  }

  DLL_PUBLIC size_t nbytes() const {
    if (!buffer_.get()) {
      return 0;
    }
    return buffer_->nbytes();
  }

  DLL_PUBLIC size_t capacity() const {
    if (!buffer_.get()) return 0;
    return buffer_->capacity();
  }

  DLL_PUBLIC Index size() const {
    return Product(shape_);
  }

  DLL_PUBLIC void set_type(TypeInfo type);

  DLL_PUBLIC void set_pinned(bool pinned) {
    // buffer_->set_pinned(pinned);
    pinned_ = pinned;
  }

  DLL_PUBLIC bool shares_data() const {
    return buffer_->shares_data();
  }

  DLL_PUBLIC Buffer<Backend> *buffer() const {
    return buffer_.get();
  }

  DLL_PUBLIC void set_num_consumers(int num) {
    num_consumers_ = num;
  }

  DLL_PUBLIC int device_id() const {
    if (!buffer_.get()) return -1;
    return buffer_->device_id();
  }

  DLL_PUBLIC void release(cudaStream_t s = nullptr) const;
  DLL_PUBLIC void force_release();

  DLL_PUBLIC void reset_reference_count() {
    reference_count_ = num_consumers_;
  }

 private:
  DLL_PUBLIC void acquire_buffer();

 protected:
  vector<Index> shape_;
  DALITensorLayout layout_;

  mutable unique_ptr<Buffer<Backend>> buffer_;

  // reference counting: on creation we will have
  // num_consumers_ operators consuming this tensor.
  // Once that many references are released we can
  // free / change the underlying Buffer storage.
  int num_consumers_;
  mutable int reference_count_;

  bool pinned_;
  bool shares_data_ = false;
  TypeInfo type_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_H_
