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

#ifndef DALI_PIPELINE_DATA_TENSOR_LIST_H_
#define DALI_PIPELINE_DATA_TENSOR_LIST_H_

#include <cstring>
#include <memory>
#include <vector>

#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"

namespace dali {

template <typename Backend>
class Tensor;

typedef vector<Index> Dims;

/**
 * @brief Stores a number of Tensors in a contiguous buffer.
 * Functions similar to a jagged tensor, i.e. a tensor
 * where each element along the outer dimension can be of
 * different size.
 *
 * Provides helper functions for accessing individual Tensors
 * in the list.
 */
template <typename Backend>
class DLL_PUBLIC TensorList {
 public:
  DLL_PUBLIC TensorList() : layout_(DALI_NHWC),
                            size_(0),
                            num_consumers_(-1),
                            pinned_(false),
                            tensor_view_(nullptr) {
    buffer_.reset();
  }

  DLL_PUBLIC ~TensorList() {
    delete tensor_view_;
  }

  /**
   * @brief Resizes this TensorList to match the shape of the input.
   */
  template <typename InBackend>
  inline void ResizeLike(const TensorList<InBackend> &other) {
    Resize(other.shape_);
  }

  /**
   * @brief Sets this TensorList to match the shape and the type
   * of the input.
   */
  template <typename InBackend>
    inline void MakeLike(const TensorList<InBackend> &other) {
      set_type_and_size(other.type(), other.shape());
    }

  /**
   * @brief Copies the input TensorList, resizing this TensorList and
   * changing the underlying data type if needed.
   */
  template <typename SrcBackend>
  DLL_PUBLIC inline void Copy(const TensorList<SrcBackend> &other, cudaStream_t stream) {
    MakeLike(other);
    type_.template Copy<Backend, SrcBackend>(raw_mutable_data(),
        other.raw_data(), size(), stream);
  }

  template <typename SrcBackend>
  DLL_PUBLIC inline void Copy(const vector<Tensor<SrcBackend>> &other, cudaStream_t stream) {
    auto type = other[0].type();

    vector<Dims> new_shape(other.size());
    for (size_t i = 0; i < other.size(); ++i) {
      DALI_ENFORCE(type == other[i].type(),
          "Cannot make a TensorList out of Tensors that do not share the common type.");
      new_shape[i] = other[i].shape();
    }

    set_type_and_size(type, new_shape);

    for (size_t i = 0; i < other.size(); ++i) {
      type_.template Copy<SrcBackend, Backend>(
          raw_mutable_tensor(i),
          other[i].raw_data(),
          other[i].size(), stream);
    }
  }

  /**
   * @brief Resize function to allocate a list of tensors. The input vector
   * contains a set of dimensions for each tensor to be allocated in the
   * list.
   */
  DLL_PUBLIC void Resize(const vector<Dims> &new_shape);

  /**
   * @brief Wraps the data owned by the input TensorList. The input
   * TensorList must have a valid type. If the input TensorList
   * stores no data, this tensor is reset to a default state
   *
   * When this function is called, the calling object shares the
   * underlying allocation of the input TensorList. Its size, type
   * and shape are set to match the calling TensorList. While this
   * list shares data with another list, 'shares_data()' will
   * return 'true'.
   */
  DLL_PUBLIC inline void ShareData(TensorList<Backend> *other) {
    buffer_.reset(new Buffer<Backend>);

    DALI_ENFORCE(other != nullptr, "Input TensorList is nullptr");
    DALI_ENFORCE(IsValidType(other->type()), "To share data, "
        "the input TensorList must have a valid data type");

    // Save the calling TensorLists meta-data
    shape_ = other->shape_;
    offsets_ = other->offsets_;
    type_ = other->type();
    size_ = other->size();

    buffer_->ShareData(other->raw_mutable_tensor(0), other->nbytes(), other->device_id());

    DALI_ENFORCE(buffer_.get() != nullptr);

    // Tensor view of this TensorList is no longer valid
    if (tensor_view_) {
      tensor_view_->ShareData(this);
    }
  }

  /**
   * @brief Wraps the raw allocation. The input pointer must not be nullptr.
   * if the size of the allocation is zero, the TensorList is reset to
   * a default state and is NOT marked as sharing data.
   *
   * After wrapping the allocation, the TensorLists size is set to 0,
   * and its type is reset to NoType. Future calls to Resize or setting
   * of the Tensor type will evaluate whether or not the current
   * allocation is large enough to be used and proceed appropriately.
   *
   * The TensorList object assumes no ownership of the input allocation,
   * and will not de-allocate it when it is done using it. It is up to
   * the user to manage the lifetime of the allocation such that it
   * persist while it is in use by the Tensor.
   */
  DLL_PUBLIC void ShareData(void *ptr, size_t bytes) {
    buffer_.reset(new Buffer<Backend>);

    DALI_ENFORCE(ptr != nullptr, "Input pointer must not be nullptr.");

    // Save our new pointer and bytes. Reset our type, shape, and size
    shape_.clear();
    offsets_.clear();

    buffer_->ShareData(ptr, bytes, -1);

    // Tensor view of this TensorList is no longer valid
    if (tensor_view_) {
      tensor_view_->ShareData(this);
    }
  }

  /**
   * @brief Wraps the raw allocation. The input pointer must not be nullptr.
   * if the size of the allocation is zero, the TensorList is reset to
   * a default state and is NOT marked as sharing data.
   *
   * This function sets the TensorList's type and shape.
   *
   * The TensorList object assumes no ownership of the input allocation,
   * and will not de-allocate it when it is done using it. It is up to
   * the user to manage the lifetime of the allocation such that it
   * persist while it is in use by the Tensor.
   */
  DLL_PUBLIC void ShareData(void *ptr, const size_t bytes,
                                   const TypeInfo &type, const vector<Dims> &shape) {
    ShareData(ptr, bytes);
    type_ = type;
    set_shape(shape);
  }

  /**
   * @brief Returns a typed pointer to the tensor with the given index.
   */
  template <typename T>
  DLL_PUBLIC inline T* mutable_tensor(int idx) {
    set_type_and_size(TypeInfo::Create<T>(), shape());
    return this->template mutable_data<T>() + tensor_offset(idx);
  }

  /**
   * @brief Returns a const typed pointer to the tensor with the given index.
   */
  template <typename T>
  DLL_PUBLIC inline const T* tensor(int idx) const {
    if (!buffer_.get()) return nullptr;
    DALI_ENFORCE(type_.id() == TypeTable::GetTypeID<T>(),
        "Calling type does not match TensorList's data type: requested " +
        TypeTable::GetTypeName<T>() + " vs stored " + type_.name());
    return buffer_->template data<T>() + tensor_offset(idx);
  }

  template <typename T>
  DLL_PUBLIC inline const T* data() const {
    if (!buffer_.get()) return nullptr;
    DALI_ENFORCE(type_.id() == TypeTable::GetTypeID<T>(),
        "Calling type does not match TensorList's data type: requested " +
        TypeTable::GetTypeName<T>() + " vs stored " + type_.name());
    return buffer_->template data<T>();
  }

  template <typename T>
  DLL_PUBLIC inline T* mutable_data() {
    set_type_and_size(TypeInfo::Create<T>(), shape());
    // catch the no-allocation case (0-byte alloc)
    if (!buffer_.get()) return nullptr;

    return buffer_->template mutable_data<T>();
  }

  /**
   * @brief Returns a raw pointer to the tensor with the given index.
   */
  DLL_PUBLIC inline void* raw_mutable_tensor(int idx) {
    acquire_buffer();
    if (!buffer_.get()) return nullptr;
    return static_cast<void*>(
        static_cast<uint8*>(this->raw_mutable_data()) +
        (tensor_offset(idx) * type_.size()));
  }

  /**
   * @brief Returns a const raw pointer to the tensor with the given index.
   */
  DLL_PUBLIC inline const void* raw_tensor(int idx) const {
    if (!buffer_.get()) return nullptr;
    return static_cast<const void*>(
        static_cast<const uint8*>(buffer_->raw_data()) +
        (tensor_offset(idx) * type_.size()));
  }

  DLL_PUBLIC inline const void *raw_data() const {
    if (!buffer_.get()) return nullptr;
    return buffer_->raw_data();
  }

  DLL_PUBLIC inline void *raw_mutable_data() {
    acquire_buffer();
    if (!buffer_.get()) return nullptr;
    return buffer_->raw_mutable_data();
  }

  /**
   * @brief Returns the number of tensors in the list.
   */
  DLL_PUBLIC inline int ntensor() const {
    return shape_.size();
  }

  /**
   * @brief Returns the offset of the tensor with the given index.
   */
  DLL_PUBLIC inline Index tensor_offset(int idx) const {
    DALI_ENFORCE(idx >= 0, "Negative index not supported");
    DALI_ENFORCE((size_t)idx < offsets_.size(), "Index out of offset range");
    return offsets_[idx];
  }

  /**
   * @brief Return the shape of the tensor with the given index.
   */
  inline vector<Index> tensor_shape(int idx) const {
    DALI_ENFORCE(idx >= 0, "Negative index not supported");
    DALI_ENFORCE((size_t)idx < shape_.size(), "Index out of offset range");
    return shape_[idx];
  }

  /**
   * @brief Returns the shape of the entire TensorList.
   */
  DLL_PUBLIC inline vector<Dims> shape() const {
    return shape_;
  }

  /**
   * @brief Checks whether the TensorList is
   * a dense Tensor. It returns true if and only if
   * all of the stored Tensors have the same shape
   * and they are densely packed in memory.
   */
  DLL_PUBLIC inline bool IsDenseTensor() const {
    if (ntensor() == 0) {
      return true;
    }
    const Dims& d = shape_[0];
    Index offset = 0;

    for (size_t i = 0; i < shape_.size(); ++i) {
      const auto& o = shape_[i];
      if (d != o) {
        return false;
      }
      if (offset != offsets_[i]) {
        return false;
      }
      offset += Product(o);
    }
    return true;
  }

  /**
   * @brief Returns a Tensor which shares the data
   * with this TensorList. The tensor obtained
   * through this function stays valid for the lifetime
   * of the parent TensorList.
   */
  Tensor<Backend> * AsTensor() {
    if (tensor_view_ == nullptr) {
      tensor_view_ = new Tensor<Backend>();
      tensor_view_->ShareData(this);
    }

    return tensor_view_;
  }

  DLL_PUBLIC TypeInfo type() const {
    return type_;
  }

  DLL_PUBLIC void set_type_and_size(TypeInfo type, const vector<Dims> &new_shape);

  DLL_PUBLIC size_t nbytes() const {
    return size() * type_.size();
  }

  DLL_PUBLIC Index size() const {
    return size_;
  }

  DLL_PUBLIC bool shares_data() const {
    if (buffer_.get() == nullptr) {
      return false;
    }
    return buffer_->shares_data();
  }

  DLL_PUBLIC void set_pinned(bool pinned) {
    pinned_ = pinned;
  }

  DLL_PUBLIC Buffer<Backend> *buffer() {
    return buffer_.get();
  }

  DLL_PUBLIC void set_num_consumers(int num) {
    num_consumers_ = num;
  }

  DLL_PUBLIC int get_num_consumers() const {
    return num_consumers_;
  }

  DLL_PUBLIC void release(cudaStream_t s = nullptr) const;
  DLL_PUBLIC void force_release();

  DLL_PUBLIC void reset_reference_count() {
    std::cout << "Resetting reference count on TL " << this << " num_consumers: " << num_consumers_ << std::endl;
    DALI_ENFORCE(num_consumers_ >= 0, "Number of consumers must be greater than 0.");
    std::cout << "Setting refcount of TL " << this << " to " << num_consumers_ << std::endl;
    reference_count_ = num_consumers_;
  }

  DLL_PUBLIC inline int device_id() const {
    if (!buffer_.get()) return -1;
    return buffer_->device_id();
  }

  // So we can access the members of other TensorListes
  // with different template types
  template <typename InBackend>
  friend class TensorList;

  DISABLE_COPY_MOVE_ASSIGN(TensorList);

  inline DALITensorLayout GetLayout() const {
    return layout_;
  }

  inline void SetLayout(DALITensorLayout layout) {
    layout_ = layout;
  }

 private:
  /**
   * @brief Acquire buffer from the global workspace.
   * After acquisition, buffer is resized to fit the
   * parent TensorList's data.
   */
  void acquire_buffer();

  /**
   * @brief Set a new shape for this TensorList.
   * This function only changes the metadata
   * and does not actually change the underlying
   * allocation.
   */
  DLL_PUBLIC void set_shape(const vector<Dims> &new_shape);

 protected:
  // We store a set of dimension for each tensor in the list.
  // We also pre-compute the offsets of each tensor in the
  // underlying allocation for random access
  vector<Dims> shape_;
  vector<Index> offsets_;
  DALITensorLayout layout_;
  Index size_;

  mutable unique_ptr<Buffer<Backend>> buffer_;

  // reference counting: on creation we will have
  // num_consumers_ operators consuming this tensor.
  // Once that many references are released we can
  // free / change the underlying Buffer storage.
  int num_consumers_;
  mutable int reference_count_;

  bool pinned_;

  TypeInfo type_;

  // In order to not leak memory (and make it slightly faster)
  // when sharing data with a Tensor, we will store a pointer to
  // Tensor that shares the data with this TensorList (valid only
  // if IsDenseTensor returns true)
  Tensor<Backend> * tensor_view_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_LIST_H_
