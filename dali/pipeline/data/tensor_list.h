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
#include "dali/pipeline/data/types.h"

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
class DLL_PUBLIC TensorList {
 public:
  DLL_PUBLIC TensorList() {}

  /**
   * @brief This constructor mostly serves as a placeholder, it should allow to get batch_size
   * nullptrs as `raw_tensor(idx)` for newly created TensorList until first proper Reshape.
   *
   * It is needed as a counterpart of TensorVector(batch_size).
   *
   * TODO(klecki): The API for empty tensor batch container of given number of samples
   * will be adjusted in next releases.
   */
  DLL_PUBLIC TensorList(int batch_size) : offsets_(batch_size, 0), meta_(batch_size) {}

  DLL_PUBLIC TensorList<Backend>(const TensorList<Backend>&) = delete;
  DLL_PUBLIC TensorList<Backend>& operator=(const TensorList<Backend>&) = delete;

  DLL_PUBLIC TensorList<Backend>(TensorList<Backend> &&other) noexcept {
    // Steal all data and set input to default state
    *this = std::move(other);
  }

  DLL_PUBLIC ~TensorList() = default;

  /**
   * @brief Number of elements in Tensor List.
   *
   * Note: The usage of this member function is intended to be reworked in following updates.
   * For this purpose the name is distinct so we can easily search and replace.
   */
  int64_t _num_elements() const {
    return data_.size();
  }

  /**
   * @brief Copies the input TensorList, resizing this TensorList and
   * changing the underlying data type if needed.
   *
   * The copy ordering can be:
   * - explict, as specified in `order`
   * - the one from `source`, if set
   * - the one from `this`
   * If neither is specified, the copy happens on the defualt stream (applies to GPU only).
   */
  template <typename SrcBackend>
  DLL_PUBLIC inline void Copy(const TensorList<SrcBackend> &other, AccessOrder order = {},
                              bool use_copy_kernel = false) {
    Resize(other.shape(), other.type());
    if (!order)
      order = other.order() ? other.order() : this->order();
    order.wait(this->order());
    this->meta_ = other.meta_;
    this->SetLayout(other.GetLayout());

    use_copy_kernel &= (std::is_same<SrcBackend, GPUBackend>::value || other.is_pinned()) &&
                       (std::is_same<Backend, GPUBackend>::value || is_pinned());
    type_info().template Copy<Backend, SrcBackend>(unsafe_raw_mutable_data(*this),
                                                   unsafe_raw_data(other),
                                                   this->_num_elements(), order.stream(),
                                                   use_copy_kernel);
    this->order().wait(order);
  }

  template <typename SrcBackend>
  DLL_PUBLIC inline void Copy(const TensorVector<SrcBackend> &other, AccessOrder order = {},
                              bool use_copy_kernel = false) {
    auto type = other[0].type();
    auto layout = other[0].GetLayout();

    int dim = other[0].shape().sample_dim();
    TensorListShape<> new_shape(other.num_samples(), dim);
    for (size_t i = 0; i < other.num_samples(); ++i) {
      DALI_ENFORCE(other[i].shape().sample_dim() == dim,
         "TensorList can only have uniform dimensions across all samples, mismatch at index "
         + std::to_string(i) + " expected Tensor with dim = " + to_string(dim)
         + " found Tensor with dim = " + to_string(other[i].shape().sample_dim()));
      assert(type == other[i].type());
      assert(layout == other[i].GetLayout());
      new_shape.set_tensor_shape(i, other[i].shape());
    }

    if (!order)
      order = other.order() ? other.order() : this->order();
    order.wait(this->order());

    this->Resize(new_shape, type);
    order.wait(this->order());
    this->SetLayout(layout);

    auto nsamples = other.num_samples();
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
                       (std::is_same<Backend, GPUBackend>::value || is_pinned());
    type_info().template Copy<SrcBackend, Backend>(dsts.data(), srcs.data(), sizes.data(),
                                                   nsamples, order.stream(), use_copy_kernel);
    this->order().wait(order);
  }

  inline void reserve(size_t bytes_per_tensor, int batch_size) {
    if (shape_.empty()) {
      offsets_.resize(batch_size, 0);
      meta_.resize(batch_size);
    }
    data_.reserve(bytes_per_tensor * batch_size);
  }

  inline void reserve(size_t bytes) {
    data_.reserve(bytes);
  }
  /**
   * @brief Resize function to allocate a list of tensors. The input vector
   * contains a set of dimensions for each tensor to be allocated in the
   * list.
   */
  DLL_PUBLIC inline void Resize(const TensorListShape<> &new_shape) {
    DALI_ENFORCE(IsValidType(type()),
                 "TensorList has no type, 'set_type<T>()' or Resize(shape, type) must be called "
                 "on the TensorList to set a valid type before it can be resized.");
    Resize(new_shape, type());
  }

  /**
   * @brief Resize function to allocate a list of tensors. The input vector
   * contains a set of dimensions for each tensor to be allocated in the
   * list.
   */
  DLL_PUBLIC inline void Resize(const TensorListShape<> &new_shape, DALIDataType new_type) {
    DALI_ENFORCE(IsValidType(new_type),
                 "TensorList cannot be resized with invalid type. To zero out the TensorList "
                 "Reset() can be used.");
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
    data_.resize(new_size, new_type);
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
  inline void ShareData(const TensorList<Backend> &other) {
    DALI_ENFORCE(IsValidType(other.type()), "To share data, "
        "the input TensorList must have a valid data type");

    // Share the underlying buffer
    data_.ShareData(other.data_);

    // Copy the shape and metadata
    shape_ = other.shape_;
    offsets_ = other.offsets_;
    meta_ = other.meta_;
    layout_ = other.layout_;

    // Tensor views of this TensorList is no longer valid
    tensor_views_.clear();
  }

  /**
   * @brief Interprets a raw allocation as a tensor list with given shape.
   *
   * If the size of the allocation is zero, the TensorList is reset to a default
   * state and is NOT marked as sharing data.
   *
   * After calling this function any following call to `set_type` and `Resize`
   * must not exceed the total size of underlying allocation (`num_bytes_`) of
   * shared data or the call will fail.
   * Size can be set to 0 and type to NoType as intermediate step.
   */
  inline void ShareData(const shared_ptr<void> &ptr, size_t bytes, bool pinned,
                        const TensorListShape<> &shape, DALIDataType type = DALI_NO_TYPE,
                        AccessOrder order = {}) {
    // Free the underlying storage.
    data_.free_storage();

    // Set the new order.
    this->set_order(order);

    // Save our new pointer and bytes. Reset our type, shape, and size
    data_.set_backing_allocation(ptr, bytes, pinned, type, shape.num_elements());
    shape_ = {};
    offsets_.clear();

    // Tensor views of this TensorList is no longer valid
    tensor_views_.clear();

    // Set the proper shape and type in one step. No-op for empty values.
    if (!shape.empty() && type != DALIDataType::DALI_NO_TYPE) {
      Resize(shape, type);
    }
  }

  /**
   * @brief Interprets a raw allocation as a tensor list with given shape.
   *
   * If the size of the allocation is zero, the TensorList is reset to a default
   * state and is NOT marked as sharing data.
   *
   * After calling this function any following call to `set_type` and `Resize`
   * must not exceed the total size of underlying allocation (`num_bytes_`) of
   * shared data or the call will fail.
   * Size can be set to 0 and type to NoType as intermediate step.
   *
   * The TensorList object assumes no ownership of the input allocation, and will
   * not de-allocate it when it is done using it. It is up to the user to
   * manage the lifetime of the allocation such that it persist while it is
   * in use by the TensorList.
   */
  DLL_PUBLIC inline void ShareData(void *ptr, size_t bytes, bool pinned,
                                   const TensorListShape<> &shape,
                                   DALIDataType type = DALI_NO_TYPE) {
    ShareData(shared_ptr<void>(ptr, [](void *) {}), bytes, pinned, shape, type);
  }

  /**
   * @brief Interprets a raw allocation as a tensor list with given shape.
   *
   * If the size of the allocation is zero, the TensorList is reset to a default
   * state and is NOT marked as sharing data.
   *
   * After calling this function any following call to `set_type` and `Resize`
   * must not exceed the total size of underlying allocation (`num_bytes_`) of
   * shared data or the call will fail.
   * Size can be set to 0 and type to NoType as intermediate step.
   *
   * The TensorList object assumes no ownership of the input allocation, and will
   * not de-allocate it when it is done using it. It is up to the user to
   * manage the lifetime of the allocation such that it persist while it is
   * in use by the TensorList.
   */
  DLL_PUBLIC inline void ShareData(void *ptr, size_t bytes, bool pinned = false,
                                   const DALIDataType type = DALI_NO_TYPE) {
    ShareData(shared_ptr<void>(ptr, [](void *) {}), bytes, pinned, TensorListShape<>{}, type);
  }

  DLL_PUBLIC void Reset(AccessOrder order = {}) {
    data_.reset(order);  // free the underlying buffer
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

      data_ = std::move(other.data_);
    }
    return *this;
  }

  /**
   * @brief TensorList is always backed by contiguous buffer
   */
  bool IsContiguous() const {
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
    return data_.template mutable_data<T>() + tensor_offset(idx);
  }

  /**
   * @brief Returns a const typed pointer to the tensor with the given index.
   */
  template <typename T>
  DLL_PUBLIC inline const T* tensor(int idx) const {
    return data_.template data<T>() + tensor_offset(idx);
  }

  /**
   * @brief Returns a raw pointer to the tensor with the given index.
   */
  DLL_PUBLIC inline void* raw_mutable_tensor(int idx) {
    return static_cast<void*>(
        static_cast<uint8*>(data_.raw_mutable_data()) +
        (tensor_offset(idx) * type_info().size()));
  }

  /**
   * @brief Returns a const raw pointer to the tensor with the given index.
   */
  DLL_PUBLIC inline const void* raw_tensor(int idx) const {
    return static_cast<const void*>(
        static_cast<const uint8*>(data_.raw_data()) +
        (tensor_offset(idx) * type_info().size()));
  }

  /**
   * @brief Returns the number of tensors in the list.
   */
  DLL_PUBLIC inline size_t num_samples() const {
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
    if (num_samples() == 0 || _num_elements() == 0) {
      return true;
    }
    if (!IsContiguous()) {
      return false;
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
    if (num_samples() == 0 || _num_elements() == 0) {
      return true;
    }
    if (!IsContiguous()) {
      return false;
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
   * @brief Returns a Tensor view with given shape or nullptr if no
   * such exists
   */
  inline Tensor<Backend> *GetViewWithShape(const TensorShape<> &shape) {
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
    DALI_ENFORCE(num_samples() > 0,
                 "To create a view Tensor, the Tensor List must have at least 1 element.");
    DALI_ENFORCE(IsValidType(type()),
                 "To create a view Tensor, the Tensor List must have a valid data type.");
    DALI_ENFORCE(IsContiguousTensor(),
                 "To create a view Tensor, all tensors in the input TensorList must be contiguous "
                 "in memory.");
    Index product = shape().num_elements();
    DALI_ENFORCE(product == volume(new_shape),
                 "To create a view Tensor, Requested shape need to have the same volume as the "
                 "tensor list.");

    tensor_views_.emplace_back();
    auto &tensor = tensor_views_.back();

    tensor.set_device_id(device_id());
    tensor.ShareData(data_.get_data_ptr(), data_.capacity(), data_.is_pinned(),
                     new_shape, type(), order());

    return &tensor;
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
    auto requested_shape = shape_cat(static_cast<int64_t>(this->num_samples()), shape_[0]);

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

  // Reexpose all public Buffer functions apart from contiguous buffer accessors.
  // TensorList is being reworked to sample-only access and this is intermediate step
  // that prevents reintroducing that access in any of DALI operators


  /**
   * @brief Returns the TypeInfo object that keeps track of the
   * datatype of the underlying storage.
   */
  const TypeInfo &type_info() const {
    return data_.type_info();
  }

  /**
   * @brief Returns the id of the datatype of the underlying storage.
   */
  DALIDataType type() const {
    return data_.type();
  }

  /**
   * @brief Returns the size in bytes of the underlying data
   */
  size_t nbytes() const {
    return data_.nbytes();
  }

  /**
   * @brief Returns the real size of the allocation
   */
  size_t capacity() const {
    return data_.capacity();
  }

  /**
   * @brief Set the type of the TensorList. The type needs to be set before calling
   * the Resize function that gives the shape. Type can be changed, if the current storage
   * is not big enough, the memory will be reallocated.
   */
  inline void set_type(const DALIDataType new_type_id) {
    data_.set_type(new_type_id);
  }

  /**
   * @brief Set the type of the TensorList. The type needs to be set before calling
   * the Resize function that gives the shape. Type can be changed, if the current storage
   * is not big enough, the memory will be reallocated.
   */
  template <typename T>
  inline void set_type() {
    data_.set_type(TypeTable::GetTypeId<T>());
  }

  /**
   * @brief Sets the type of allocation (pinned/non-pinned) for CPU TensorList
   */
  inline void set_pinned(bool pinned) {
    data_.set_pinned(pinned);
  }

  /**
   * @brief Returns the type of allocation (pinned/non-pinned) for CPU TensorList
   */
  bool is_pinned() const {
    return data_.is_pinned();
  }

   /**
   * @brief Returns a device this TensorList was allocated on
   * If the backend is CPUBackend, return -1
   */
  int device_id() const {
    return data_.device_id();
  }

  /**
   * @brief Sets a device this TensorList was allocated on
   * If the backend is CPUBackend, should be -1
   */
  void set_device_id(int device) {
    data_.set_device_id(device);
  }

  /**
   * @brief Returns the order in which the data is accessed - it can be either host order
   *        or a stream order (or unspecified).
   */
  AccessOrder order() const {
    return data_.order();
  }

  /**
   * @brief Sets the associated access order.
   *
   * @note The caller must ensure that if `order` represents a CUDA stream, that stream
   *       is alive when this buffer is destroyed. This extends to buffers with which this
   *       one shares data. Use CUDAStreamPool::instance to get streams with indefinite lifetime.
   *
   * @param order       The new access order (stream or host). If the new order doesn't have
   *                    a value, the function has no effect.
   * @param synchronize If true, an appropriate synchronization is inserted between the old
   *                    and the new order. The caller may specify `false` if appropriate
   *                    synchronization is guaranteed by other means.
   */
  void set_order(AccessOrder order, bool synchronize = true) {
    data_.set_order(order);
  }

  /**
   * @brief Return true if there was data allocation
   */
  inline bool has_data() const noexcept {
    return data_.has_data();
  }

  /**
   * @brief Returns a bool indicating if the list shares its underlying storage.
   */
  inline bool shares_data() const {
    return data_.shares_data();
  }

  /**
   * @brief Sets a custom allocation function.
   *
   * Sets a custom allocation function. The allocation function returns
   * a shared pointer with a matching deleter.
   *
   * @remarks Experimental - subject to change
   */
  inline void set_alloc_func(typename Buffer<Backend>::AllocFunc allocate) {
    data_.set_alloc_func(std::move(allocate));
  }

  /**
   * @brief Returns the current custom allocation function.
   *
   * @return Allocation function. If not set, an empty function object is returned.
   *
   * @remarks Experimental - subject to change
   */
  const typename Buffer<Backend>::AllocFunc &alloc_func() const noexcept {
    return data_.alloc_func();
  }

 protected:
  Buffer<Backend> data_ = {};
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
  std::list<Tensor<Backend>> tensor_views_;

 private:
  /** @defgroup ContiguousAccessorFunctions Fallback contiguous accessors
   * Fallback access to contiguous data to TensorList. It should not be used for processing,
   * and can be used only for outputs of the pipeline that were made sure to be contiguous.
   * Currently TensorList is contiguous by design, but it is up to change.
   * @{
   */

  /**
   * @brief Return an un-typed pointer to the underlying storage.
   * The TensorList must be either empty or have a valid type and be contiguous.
   */
  friend void *unsafe_raw_mutable_data(TensorList<Backend> &tl) {
    DALI_ENFORCE(tl.IsContiguous(), "Data pointer can be obtain only for contiguous TensorList.");
    return tl.data_.raw_mutable_data();
  }

  /**
   * @brief Return an un-typed const pointer to the underlying storage.
   * The TensorList must be either empty or have a valid type and be contiguous.
   */
  friend const void *unsafe_raw_data(const TensorList<Backend> &tl) {
    DALI_ENFORCE(tl.IsContiguous(), "Data pointer can be obtain only for contiguous TensorList.");
    return tl.data_.raw_data();
  }

  /**
   * @brief Return the shared pointer, that we can use to correctly share the ownership of sample
   * with.
   */
  friend shared_ptr<void> unsafe_sample_owner(TensorList<Backend> &tl, int sample_idx) {
    // create new aliasing pointer to current data allocation, so we share the use count
    // and the deleter correctly.
    return {tl.data_.get_data_ptr(), tl.raw_mutable_tensor(sample_idx)};
  }

  /** @} */  // end of ContiguousAccessorFunctions
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_LIST_H_
