// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/pipeline/data/tensor_list.h"

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
    // Point to data, as we are sharing use no-op deleter
    view.data_.reset(static_cast<uint8_t *>(this->raw_mutable_data()) + x * view.num_bytes_,
                     [](void *) {});
    view.shares_data_ = true;
    view.device_ = device_;
    return view;
  }

  /**
   * Loads the Tensor with data from the input vector.
   */
  template <typename T>
  inline void Copy(const vector<T> &data, cudaStream_t stream) {
    this->template mutable_data<T>();
    this->Resize({(Index)data.size()});
    type_.template Copy<Backend, CPUBackend>(this->raw_mutable_data(),
        data.data(), this->size(), stream);
  }

  /**
   * Loads the Tensor with data from a span.
   */
  template <typename T>
  inline void Copy(span<T> data, cudaStream_t stream) {
    using U = remove_const_t<T>;
    this->template mutable_data<U>();
    this->Resize({(Index)data.size()});
    type_.template Copy<Backend, CPUBackend>(this->raw_mutable_data(),
        data.data(), this->size(), stream);
  }

  /**
   * Loads the Tensor with data from the input Tensor.
   */
  template <typename InBackend>
  inline void Copy(const Tensor<InBackend> &other, cudaStream_t stream) {
    this->set_type(other.type());
    this->SetLayout(other.GetLayout());
    this->SetSourceInfo(other.GetSourceInfo());
    this->SetSkipSample(other.ShouldSkipSample());
    this->ResizeLike(other);
    type_.template Copy<Backend, InBackend>(this->raw_mutable_data(),
        other.raw_data(), this->size(), stream);
  }

  /**
   * @brief Loads the Tensor at index idx from the input TensorList.
   */
  template <typename InBackend>
  inline void Copy(const TensorList<InBackend> &other, int idx, cudaStream_t stream) {
    shape_ = other.tensor_shape(idx);
    device_ = other.device_id();
    this->set_type(other.type());
    this->SetLayout(other.GetLayout());
    this->SetSourceInfo(other.GetSourceInfo(idx));
    this->SetSkipSample(other.ShouldSkipSample(idx));
    this->Resize(shape_);
    type_.template Copy<Backend, InBackend>(this->raw_mutable_data(),
        other.raw_tensor(idx), this->size(), stream);
  }

  template <typename InBackend>
  inline void ResizeLike(const Tensor<InBackend> &other) {
    Resize(other.shape());
  }

  /**
   * @brief Resizes the buffer to fit `volume(shape)` elements.
   * The underlying storage is only reallocated in the case that
   * the current buffer is not large enough for the requested
   * number of elements.
   */
  inline void Resize(const TensorShape<> &shape) {
    Index new_size = volume(shape);
    ResizeHelper(new_size);
    shape_ = shape;
  }

  /**
   * @brief Resizes the buffer to fit `volume(shape)` of new_type elements.
   * The underlying storage is only reallocated in the case that
   * the current buffer is not large enough for the requested
   * number of elements.
   */
  inline void Resize(const TensorShape<> &shape, const TypeInfo &new_type) {
    Index new_size = volume(shape);
    ResizeHelper(new_size, new_type);
    shape_ = shape;
  }

  /**
   * @brief Tensor is always backed by contiguous buffer
   */
  bool IsContiguous() {
    return true;
  }

  /**
   * @brief Tensor is always backed by contiguous buffer
   *        Cannot be set to noncontiguous
   */
  void SetContiguous(bool contiguous) {
    DALI_ENFORCE(contiguous, "Tensor cannot be made noncontiguous");
  }

  using Buffer<Backend>::reserve;

  // For having complete API, Tensor is not a batch
  void reserve(size_t bytes_per_tensor, int) {
    reserve(bytes_per_tensor);
  }

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
   *
   * After calling this function any following call to `set_type` and `Resize`
   * must match the total size of underlying allocation (`num_bytes_`) of
   * shared data or the call will fail.
   * Size can be set to 0 and type to NoType as intermediate step.
   */
  inline void ShareData(TensorList<Backend> *tl, int idx) {
    DALI_ENFORCE(tl != nullptr, "Input TensorList is nullptr");
    DALI_ENFORCE(IsValidType(tl->type()), "To share data, "
        "the input TensorList must have a valid data type.");
    DALI_ENFORCE(idx >= 0, "Negative tensor index not supported.");
    DALI_ENFORCE(static_cast<size_t>(idx) < tl->ntensor(), "Index of " + std::to_string(idx) +
        " out of range for TensorList of size " + std::to_string(tl->ntensor()));

    // Reset our pointer to the correct offset inside the tensor list.
    // This is not the beginning of the allocation, so we pass a noop
    // deleter to the shared_ptr
    data_.reset(tl->raw_mutable_tensor(idx), [](void *) {});

    // Get the meta-data for the target tensor
    shape_ = tl->tensor_shape(idx);
    size_ = volume(shape_);
    type_ = tl->type();
    num_bytes_ = type_.size() * size_;
    shares_data_ = true;
    device_ = tl->device_id();
    meta_ = tl->GetMeta(idx);
  }

  /**
   * @brief Wraps the data owned by the input tensor. The input
   * tensor must have a valid type. If sucessful, the tensor
   * object will wrap the target data and assume the datatype
   * and shape of the data stored in the Tensor.
   *
   * If the input does not store any data, shares_data_ is left
   * as false.
   *
   * After calling this function any following call to `set_type` and `Resize`
   * must match the total size of underlying allocation (`num_bytes_`) of
   * shared data or the call will fail.
   * Size can be set to 0 and type to NoType as intermediate step.
   */
  inline void ShareData(Tensor<Backend> *t) {
    DALI_ENFORCE(t != nullptr, "Input Tensor is nullptr");
    DALI_ENFORCE(IsValidType(t->type()), "To share data, "
        "the input Tensor must have a valid data type.");

    // Save a copy of our new data pointer. We create a copy of the
    // shared_ptr to ensure the data persists while we are still
    // using it.
    data_ = t->data_;

    // Save the tensor meta-data
    shape_ = t->shape_;
    size_ = t->size_;
    type_ = t->type_;
    num_bytes_ = t->num_bytes_;
    shares_data_ = num_bytes_ > 0 ? true : false;
    device_ = t->device_id();
    meta_ = t->meta_;
  }

  /**
   * @brief Wraps the raw allocation. The input pointer must not be nullptr.
   * if the size of the allocation is zero, the Tensor is reset to a default
   * state and is NOT marked as sharing data. Also sets shape of new Tensor.
   *
   * After wrapping the allocation, the Tensors size is set to dot product
   * of shape vector, and its type is reset to NoType.
   * After calling this function any following call to `set_type` and `Resize`
   * must match the total size of underlying allocation (`num_bytes_`) of
   * shared data or the call will fail.
   * Size can be set to 0 and type to NoType as intermediate step.
   *
   * The Tensor object assumes no ownership of the input allocation, and will
   * not de-allocate it when it is done using it. It is up to the user to
   * manage the lifetime of the allocation such that it persist while it is
   * in use by the Tensor.
   */
  inline void ShareData(const shared_ptr<void> &ptr, size_t bytes,
                        const TensorShape<> &shape,
                        const TypeInfo &type = {}) {
    // don't check ptr as we want to share empty data as well

    // Save our new pointer and bytes. Reset our type, shape, and size
    data_ = ptr;
    num_bytes_ = bytes;
    type_ = type;
    Index new_size = volume(shape);
    shape_ = shape;
    size_ = new_size;

    // If the input pointer stores a non-zero size allocation, mark
    // that we are sharing our underlying data
    shares_data_ = num_bytes_ > 0 ? true : false;
  }

  /**
   * @brief Wraps the raw allocation. The input pointer must not be nullptr.
   * if the size of the allocation is zero, the Tensor is reset to a default
   * state and is NOT marked as sharing data. Also sets shape of new Tensor.
   *
   * After wrapping the allocation, the Tensors size is set to dot product
   * of shape vector, and its type is reset to NoType.
   * After calling this function any following call to `set_type` and `Resize`
   * must match the total size of underlying allocation (`num_bytes_`) of
   * shared data or the call will fail.
   * Size can be set to 0 and type to NoType as intermediate step.
   *
   * The Tensor object assumes no ownership of the input allocation, and will
   * not de-allocate it when it is done using it. It is up to the user to
   * manage the lifetime of the allocation such that it persist while it is
   * in use by the Tensor.
   */
  inline void ShareData(void *ptr, size_t bytes, const TensorShape<> &shape,
                        const TypeInfo &type = TypeInfo::Create<NoType>()) {
    ShareData(shared_ptr<void>(ptr, [](void *) {}), bytes, shape, type);
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
   */
  inline void ShareData(void *ptr, size_t bytes,
                        const TypeInfo &type = TypeInfo::Create<NoType>()) {
    ShareData(ptr, bytes, { 0 }, type);
  }

  /**
   * @brief Wraps a TensorList and gives it a new shape
   * TensorList has to be a valid tensor
   * (there must be at least 1 tensor stored in the TensorList,
   * volumes of the new and old shape need to match and
   * all tensors need to be stored without
   * any padding between them)
   */
  inline void ShareDataReshape(TensorList<Backend> *tl, const TensorShape<> &new_shape) {
    DALI_ENFORCE(tl != nullptr, "Input TensorList is nullptr");
    DALI_ENFORCE(tl->ntensor() > 0, "Input TensorList has 0 elements!");
    DALI_ENFORCE(IsValidType(tl->type()), "To share data, "
        "the input TensorList must have a valid data type.");
    DALI_ENFORCE(tl->IsContiguousTensor(),
      "All tensors in the input TensorList must be contiguous in memory.");
    Index product = tl->shape().num_elements();
    DALI_ENFORCE(product == volume(new_shape),
      "Requested shape need to have the same volume as the tensor list.");
    data_.reset(tl->raw_mutable_tensor(0), [](void *) {});

    // Get the meta-data for the target tensor
    shape_ = new_shape;
    size_ = volume(shape_);
    type_ = tl->type();
    num_bytes_ = type_.size() * size_;
    device_ = tl->device_id();
    shares_data_ = true;
    meta_ = {};
  }

  /**
   * @brief Wraps a TensorList
   * TensorList has to be a valid tensor
   * (there must be at least 1 tensor stored in TensorList,
   * all shapes should be identical,
   * all tensors need to be stored without
   * any offset between them)
   */
  inline void ShareData(TensorList<Backend> *tl) {
    DALI_ENFORCE(tl != nullptr, "Input TensorList is nullptr");
    DALI_ENFORCE(tl->ntensor() > 0, "Input TensorList has 0 elements!");
    DALI_ENFORCE(IsValidType(tl->type()), "To share data, "
        "the input TensorList must have a valid data type.");
    DALI_ENFORCE(tl->IsDenseTensor(),
      "All tensors in the input TensorList must have the same shape and be densely packed.");
    data_.reset(tl->raw_mutable_tensor(0), [](void *) {});

    // Get the meta-data for the target tensor
    shape_ = shape_cat(tl->ntensor(), tl->tensor_shape(0));
    size_ = volume(shape_);
    type_ = tl->type();
    num_bytes_ = type_.size() * size_;
    device_ = tl->device_id();
    shares_data_ = true;
    if (!tl->GetLayout().empty())
      SetLayout("N" + tl->GetLayout());
    else
      SetLayout({});
  }

  inline void Reset() {
    reset();  // free the underlying buffer
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

  Tensor<Backend>(const Tensor<Backend>&) = delete;
  Tensor<Backend>& operator=(const Tensor<Backend>&) = delete;

  Tensor<Backend>(Tensor<Backend> &&t) noexcept {
    // Steal all data and set input to default state
    shape_ = std::move(t.shape_);
    meta_ = std::move(t.meta_);

    t.shape_ = TensorShape<>();
    t.meta_ = {};
    move_buffer(std::move(t));
  }

  Tensor<Backend>& operator=(Tensor<Backend> &&t) noexcept {
    if (&t != this) {
      shape_ = std::move(t.shape_);
      meta_ = std::move(t.meta_);

      t.shape_ = TensorShape<>();
      t.meta_ = {};
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

  // So TensorVector can access data_ of the tensor directly
  template <typename InBackend>
  friend class TensorVector;
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_H_
