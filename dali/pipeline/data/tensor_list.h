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

#ifndef DALI_PIPELINE_DATA_TENSOR_LIST_H_
#define DALI_PIPELINE_DATA_TENSOR_LIST_H_

#include <assert.h>
#include <cstring>
#include <string>
#include <vector>
#include <list>
#include <memory>
#include <utility>
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/meta.h"

namespace dali {

template <typename Backend>
class Tensor;

template <typename Backend>
class TensorVector;

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
class DLL_PUBLIC TensorList : public Buffer<Backend> {
 public:
  DLL_PUBLIC TensorList() {}

  DLL_PUBLIC TensorList(int batch_size) {
    Resize(TensorListShape<>(batch_size));
  }

  DLL_PUBLIC TensorList<Backend>(const TensorList<Backend>&) = delete;
  DLL_PUBLIC TensorList<Backend>& operator=(const TensorList<Backend>&) = delete;

  DLL_PUBLIC TensorList<Backend>(TensorList<Backend> &&other) noexcept {
    // Steal all data and set input to default state
    *this = std::move(other);
  }

  DLL_PUBLIC ~TensorList() = default;

  /**
   * @brief Resizes this TensorList to match the shape of the input.
   */
  template <typename InBackend>
  inline void ResizeLike(const TensorList<InBackend> &other) {
    Resize(other.shape_);
  }

  /**
   * @brief Copies the input TensorList, resizing this TensorList and
   * changing the underlying data type if needed.
   */
  template <typename SrcBackend>
  DLL_PUBLIC inline void Copy(const TensorList<SrcBackend> &other, cudaStream_t stream,
                              bool use_copy_kernel = false) {
    if (IsValidType(other.type())) {
      this->set_type(other.type());
    }
    this->meta_ = other.meta_;
    this->SetLayout(other.GetLayout());
    ResizeLike(other);

    use_copy_kernel &= (std::is_same<SrcBackend, GPUBackend>::value || other.is_pinned()) &&
                       (std::is_same<Backend, GPUBackend>::value || pinned_);
    type_.template Copy<Backend, SrcBackend>(this->raw_mutable_data(), other.raw_data(),
                                             this->size(), stream, use_copy_kernel);
  }

  template <typename SrcBackend>
  DLL_PUBLIC inline void Copy(const TensorVector<SrcBackend> &other, cudaStream_t stream,
                              bool use_copy_kernel = false) {
    auto type = other[0].type();
    auto layout = other[0].GetLayout();

    int dim = other[0].shape().sample_dim();
    TensorListShape<> new_shape(other.size(), dim);
    for (size_t i = 0; i < other.size(); ++i) {
      DALI_ENFORCE(other[i].shape().sample_dim() == dim,
         "TensorList can only have uniform dimensions across all samples, mismatch at index "
         + std::to_string(i) + " expected Tensor with dim = " + to_string(dim)
         + " found Tensor with dim = " + to_string(other[i].shape().sample_dim()));
      assert(type == other[i].type());
      assert(layout == other[i].GetLayout());
      new_shape.set_tensor_shape(i, other[i].shape());
    }

    this->Resize(new_shape);
    if (IsValidType(type)) {
      this->set_type(type);
    }
    this->SetLayout(layout);

    auto nsamples = other.size();
    SmallVector<const void*, 256> srcs;
    srcs.reserve(nsamples);
    SmallVector<void*, 256> dsts;
    dsts.reserve(nsamples);
    SmallVector<Index, 256> sizes;
    sizes.reserve(nsamples);
    for (size_t i = 0; i < nsamples; i++) {
      dsts.emplace_back(this->raw_mutable_tensor(i));
      srcs.emplace_back(other[i].raw_data());
      sizes.emplace_back(other[i].size());
      this->meta_[i].SetSourceInfo(other[i].GetSourceInfo());
      this->meta_[i].SetSkipSample(other[i].ShouldSkipSample());
    }

    use_copy_kernel &= (std::is_same<SrcBackend, GPUBackend>::value || other.is_pinned()) &&
                       (std::is_same<Backend, GPUBackend>::value || pinned_);
    type.template Copy<SrcBackend, Backend>(dsts.data(), srcs.data(), sizes.data(),
                                            nsamples, stream, use_copy_kernel);
  }

  using Buffer<Backend>::reserve;

  inline void reserve(size_t bytes_per_tensor, int batch_size) {
    if (shape_.empty()) {
      Resize(TensorListShape<>(batch_size));
    }
    reserve(bytes_per_tensor * batch_size);
  }

  /**
   * @brief Resize function to allocate a list of tensors. The input vector
   * contains a set of dimensions for each tensor to be allocated in the
   * list.
   */
  DLL_PUBLIC inline void Resize(const TensorListShape<> &new_shape) {
    Resize(new_shape, type_);
  }

  /**
   * @brief Resize function to allocate a list of tensors. The input vector
   * contains a set of dimensions for each tensor to be allocated in the
   * list.
   */
  DLL_PUBLIC inline void Resize(const TensorListShape<> &new_shape, const TypeInfo &new_type) {
    // Calculate the new size
    Index num_tensor = new_shape.size(), new_size = 0;
    offsets_.resize(num_tensor);
    for (Index i = 0; i < num_tensor; ++i) {
      auto tensor_size = volume(new_shape[i]);

      // Save the offset of the current sample & accumulate the size
      offsets_[i] = new_size;
      new_size += tensor_size;
    }
    DALI_ENFORCE(new_size >= 0, "Invalid negative buffer size.");

    // Resize the underlying allocation and save the new shape
    ResizeHelper(new_size, new_type);
    shape_ = new_shape;

    // Tensor views of this TensorList is no longer valid
    tensor_views_.clear();

    meta_.resize(num_tensor, DALIMeta(layout_));
  }

  /**
   * @brief Wraps the data owned by the input TensorList. The input
   * TensorList must have a valid type. If the input TensorList
   * stores no data, this tensor is reset to a default state.
   *
   * When this function is called, the calling object shares the
   * underlying allocation of the input TensorList. Its size, type
   * and shape are set to match the calling TensorList. While this
   * list shares data with another list, 'shares_data()' will
   * return 'true'.
   *
   * After calling this function any following call to `set_type` and `Resize`
   * must match the total size of underlying allocation (`num_bytes_`) of
   * shared data or the call will fail.
   * Size can be set to 0 and type to NoType as intermediate step.
   */
  DLL_PUBLIC inline void ShareData(TensorList<Backend> *other) {
    DALI_ENFORCE(other != nullptr, "Input TensorList is nullptr");
    DALI_ENFORCE(IsValidType(other->type_), "To share data, "
        "the input TensorList must have a valid data type");

    // Save the calling TensorLists meta-data
    data_ = other->data_;
    shape_ = other->shape_;
    size_ = other->size_;
    offsets_ = other->offsets_;
    type_ = other->type_;
    num_bytes_ = other->num_bytes_;
    device_ = other->device_;

    // Tensor views of this TensorList is no longer valid
    tensor_views_.clear();

    // If the other tensor has a non-zero size allocation, mark that
    // we are now sharing an allocation with another buffer
    shares_data_ = num_bytes_ > 0 ? true : false;

    // copy metadata
    meta_ = other->meta_;
    layout_ = other->layout_;
  }

  /**
   * @brief Wraps the raw allocation. The input pointer must not be nullptr.
   * if the size of the allocation is zero, the TensorList is reset to
   * a default state and is NOT marked as sharing data.
   *
   * The size of the tensor list is calculated based on shape and type or reset to 0
   * if the shape is empty or the type is DALI_NO_TYPE.
   * After calling this function any following call to `set_type` and `Resize`
   * must match the total size of underlying allocation (`num_bytes_`) of
   * shared data or the call will fail.
   * Size can be set to 0 and type to NoType as intermediate step.
   *
   * The TensorList object assumes no ownership of the input allocation,
   * and will not de-allocate it when it is done using it. It is up to
   * the user to manage the lifetime of the allocation such that it
   * persist while it is in use by the Tensor.
   */
  inline void ShareData(const shared_ptr<void> &ptr, size_t bytes, const TensorListShape<> &shape,
                        const TypeInfo &type = {}) {
    // don't check ptr as we want to share empty data as well

    // Save our new pointer and bytes. Reset our type, shape, and size
    data_ = ptr;
    num_bytes_ = bytes;
    type_ = type;
    shape_ = {};
    offsets_.clear();
    size_ = 0;
    device_ = CPU_ONLY_DEVICE_ID;

    // Tensor views of this TensorList is no longer valid
    tensor_views_.clear();

    // If the input pointer stores a non-zero size allocation, mark
    // that we are sharing our underlying data
    shares_data_ = num_bytes_ > 0 ? true : false;
    // Set the proper shape and type in one step. No-op for empty values.
    if (!shape.empty() && type.id() != DALIDataType::DALI_NO_TYPE) {
      Resize(shape, type);
    }
  }

  /**
   * @brief Wraps the raw allocation. The input pointer must not be nullptr.
   * if the size of the allocation is zero, the TensorList is reset to
   * a default state and is NOT marked as sharing data.
   *
   * The size of the tensor list is calculated based on shape and type or reset to 0
   * if the shape is empty or the type is DALI_NO_TYPE.
   * After calling this function any following call to `set_type` and `Resize`
   * must match the total size of underlying allocation (`num_bytes_`) of
   * shared data or the call will fail.
   * Size can be set to 0 and type to NoType as intermediate step.
   *
   * The TensorList object assumes no ownership of the input allocation,
   * and will not de-allocate it when it is done using it. It is up to
   * the user to manage the lifetime of the allocation such that it
   * persist while it is in use by the Tensor.
   */
  DLL_PUBLIC inline void ShareData(void *ptr, size_t bytes, const TensorListShape<> &shape,
                                   const TypeInfo &type = {}) {
    ShareData(shared_ptr<void>(ptr, [](void *) {}), bytes, shape, type);
  }

  /**
   * @brief Wraps the raw allocation. The input pointer must not be nullptr.
   * if the size of the allocation is zero, the TensorList is reset to
   * a default state and is NOT marked as sharing data.
   *
   * After wrapping the allocation, the TensorLists size is set to 0,
   * and its type is reset to NoType (if not provided otherwise).
   * After calling this function any following call to `set_type` and `Resize`
   * must match the total size of underlying allocation (`num_bytes_`) of
   * shared data or the call will fail.
   * Size can be set to 0 and type to NoType as intermediate step.
   *
   * The TensorList object assumes no ownership of the input allocation,
   * and will not de-allocate it when it is done using it. It is up to
   * the user to manage the lifetime of the allocation such that it
   * persist while it is in use by the Tensor.
   */
  DLL_PUBLIC inline void ShareData(void *ptr, size_t bytes,
                                   const TypeInfo &type = TypeInfo::Create<NoType>()) {
    ShareData(shared_ptr<void>(ptr, [](void *) {}), bytes, TensorListShape<>{}, type);
  }

  DLL_PUBLIC void Reset() {
    reset();  // free the underlying buffer
    shape_ = {};
    offsets_.clear();
    meta_.clear();
    tensor_views_.clear();
  }

  DLL_PUBLIC inline TensorList<Backend>& operator=(TensorList<Backend> &&other) noexcept {
    if (&other != this) {
      shape_ = std::move(other.shape_);
      offsets_ = std::move(other.offsets_);
      tensor_views_ = std::move(other.tensor_views_);
      meta_ = std::move(other.meta_);
      layout_ = std::move(other.layout_);

      other.shape_ = {};
      other.tensor_views_.clear();
      other.offsets_.clear();
      other.meta_.clear();
      other.layout_ = {};

      move_buffer(std::move(other));
    }
    return *this;
  }

  /**
   * @brief TensorList is always backed by contiguous buffer
   */
  bool IsContiguous() {
    return true;
  }

  /**
   * @brief TensorList is always backed by contiguous buffer
   *        Cannot be set to noncontiguous
   */
  void SetContiguous(bool contiguous) {
    DALI_ENFORCE(contiguous, "TensorList cannot be made noncontiguous");
  }

  /**
   * @brief Returns a typed pointer to the tensor with the given index.
   */
  template <typename T>
  DLL_PUBLIC inline T* mutable_tensor(int idx) {
    return this->template mutable_data<T>() + tensor_offset(idx);
  }

  /**
   * @brief Returns a const typed pointer to the tensor with the given index.
   */
  template <typename T>
  DLL_PUBLIC inline const T* tensor(int idx) const {
    return this->template data<T>() + tensor_offset(idx);
  }

  /**
   * @brief Returns a raw pointer to the tensor with the given index.
   */
  DLL_PUBLIC inline void* raw_mutable_tensor(int idx) {
    return static_cast<void*>(
        static_cast<uint8*>(this->raw_mutable_data()) +
        (tensor_offset(idx) * type_.size()));
  }

  /**
   * @brief Returns a const raw pointer to the tensor with the given index.
   */
  DLL_PUBLIC inline const void* raw_tensor(int idx) const {
    return static_cast<const void*>(
        static_cast<const uint8*>(this->raw_data()) +
        (tensor_offset(idx) * type_.size()));
  }

  /**
   * @brief Returns the number of tensors in the list.
   */
  DLL_PUBLIC inline size_t ntensor() const {
    return shape_.size();
  }

  /**
   * @brief Returns the number of dimensions of the samples.
   */
  DLL_PUBLIC inline int sample_dim() const {
    return shape_.sample_dim();
  }


  /**
   * @brief Returns the offset of the tensor with the given index.
   */
  DLL_PUBLIC inline Index tensor_offset(int idx) const {
#ifndef NDEBUG
    DALI_ENFORCE(idx >= 0, "Negative index not supported");
    DALI_ENFORCE((size_t)idx < offsets_.size(), "Index out of offset range");
#endif
    return offsets_[idx];
  }

  /**
   * @brief Return the shape of the tensor with the given index.
   */
  inline TensorShape<> tensor_shape(int idx) const {
#ifndef NDEBUG
    DALI_ENFORCE(idx >= 0, "Negative index not supported");
    DALI_ENFORCE(idx < shape_.size(), "Index out of offset range");
#endif
    return shape_[idx];
  }

  /**
   * @brief Return the shape of the tensor with the given index.
   */
  inline span<const int64_t> tensor_shape_span(int idx) const {
#ifndef NDEBUG
    DALI_ENFORCE(idx >= 0, "Negative index not supported");
    DALI_ENFORCE(idx < shape_.size(), "Index out of offset range");
#endif
    return shape_.tensor_shape_span(idx);
  }

  /**
   * @brief Returns the shape of the entire TensorList.
   */
  inline const TensorListShape<> &shape() const {
    return shape_;
  }

  /**
   * @brief Checks whether the TensorList is
   * contiguous. It returns true if and only if
   * all of the stored Tensors are densely packed in memory.
   */
  inline bool IsContiguousTensor() const {
    if (ntensor() == 0 || size_ == 0) {
      return true;
    }
    Index offset = 0;

    for (int i = 0; i < shape_.size(); ++i) {
      if (offset != offsets_[i]) {
        return false;
      }
      offset += volume(shape_[i]);
    }
    return true;
  }

  /**
   * @brief Checks whether the TensorList is
   * a dense Tensor. It returns true if and only if
   * all of the stored Tensors have the same shape
   * and they are densely packed in memory.
   */
  inline bool IsDenseTensor() const {
    if (ntensor() == 0 || size_ == 0) {
      return true;
    }
    if (!is_uniform(shape_)) {
      return false;
    }
    // shapes are uniform, check if offsets are packed
    auto tensor_volume = volume(shape_[0]);
    Index offset = 0;

    for (int i = 0; i < shape_.size(); ++i) {
      if (offset != offsets_[i]) {
        return false;
      }
      offset += tensor_volume;
    }
    return true;
  }

  /**
   * @brief Returns the number of elements
   *  in the TensorList
   */
  inline size_t GetElementsNumber() const {
    return shape_.num_elements();
  }

  /**
   * @brief Returns a Tensor view with given shape or nullptr if no
   * such exists
   */
  inline Tensor<Backend> * GetViewWithShape(const TensorShape<> &shape) {
    for (auto &t : tensor_views_) {
      if (t.shape() == shape) {
        return &t;
      }
    }
    return nullptr;
  }

  /**
   * @brief Returns a pointer to Tensor which shares the data
   * with this TensorList and give it the provided shape.
   * Tensor list owns the memory. The tensor obtained through
   * this function stays valid for as long as TensorList data is unchanged.
   */
  DLL_PUBLIC inline Tensor<Backend> * AsReshapedTensor(const TensorShape<> &new_shape) {
    auto t = GetViewWithShape(new_shape);
    if (t) {
      return t;
    }

    // need to create a new view
    tensor_views_.emplace_back();
    tensor_views_.back().ShareDataReshape(this, new_shape);

    return &tensor_views_.back();
  }

  /**
   * @brief Returns a pointer to Tensor which shares the data
   * with this TensorList. Tensor list owns the memory. The tensor
   * obtained through this function stays valid for as long
   * as TensorList data is unchanged.
   */
  DLL_PUBLIC inline Tensor<Backend> * AsTensor() {
    // To prevent situation when AsReshapedTensor is called first with some shape, and then
    // AsTensor which return non-dense tensor after all
    // i.e. [[2], [3], [1]] is not dense but requesting [3, 2] AsReshapedTensor will work
    // while AsTensor should not return for that case
    DALI_ENFORCE(this->IsDenseTensor(),
      "All tensors in the input TensorList must have the same shape and be densely packed.");
    auto requested_shape = shape_cat(static_cast<int64_t>(this->ntensor()), shape_[0]);

    return this->AsReshapedTensor(requested_shape);
  }


  // So we can access the members of other TensorListes
  // with different template types
  template <typename InBackend>
  friend class TensorList;

  inline std::string GetSourceInfo(int idx) const {
    return meta_[idx].GetSourceInfo();
  }

  inline void SetSourceInfo(int idx, const std::string& source_info) {
    meta_[idx].SetSourceInfo(source_info);
  }

  inline TensorLayout GetLayout() const {
    // Layout is enforced to be the same across all the samples
    return layout_;
  }

  /** @brief Set uniform layout for all samples in the list */
  inline void SetLayout(const TensorLayout &layout) {
    layout_ = layout;
    for (auto& meta : meta_)
      meta.SetLayout(layout);
  }

  inline void SetSkipSample(int idx, bool skip_sample) {
    return meta_[idx].SetSkipSample(skip_sample);
  }

  inline bool ShouldSkipSample(int idx) const {
    return meta_[idx].ShouldSkipSample();
  }

  inline const DALIMeta &GetMeta(int idx) const {
    return meta_[idx];
  }

  inline void SetMeta(int idx, const DALIMeta &meta) {
    meta_[idx] = meta;
  }

 protected:
  // We store a set of dimension for each tensor in the list.
  // We also pre-compute the offsets of each tensor in the
  // underlying allocation for random access
  TensorListShape<> shape_;
  vector<Index> offsets_;
  vector<DALIMeta> meta_;
  TensorLayout layout_;

  // In order to not leak memory (and make it slightly faster)
  // when sharing data with a Tensor, we will store a pointer to
  // Tensor that shares the data with this TensorList (valid only
  // if IsDenseTensor returns true)
  std::list<Tensor<Backend> > tensor_views_;

  USE_BUFFER_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_LIST_H_
