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

#include <assert.h>
#include <cstring>
#include <string>
#include <vector>
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/meta.h"

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
class DLL_PUBLIC TensorList : public Buffer<Backend> {
 public:
  DLL_PUBLIC TensorList()
    : layout_(DALI_NHWC)
    , tensor_view_(nullptr) {}

  DLL_PUBLIC ~TensorList() override {
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
   * @brief Copies the input TensorList, resizing this TensorList and
   * changing the underlying data type if needed.
   */
  template <typename SrcBackend>
  DLL_PUBLIC inline void Copy(const TensorList<SrcBackend> &other, cudaStream_t stream) {
    if (IsValidType(other.type())) {
      this->set_type(other.type());
    }
    this->meta_ = other.meta_;
    this->SetLayout(other.GetLayout());
    ResizeLike(other);
    type_.template Copy<Backend, SrcBackend>(this->raw_mutable_data(),
        other.raw_data(), this->size(), stream);
  }

  template <typename SrcBackend>
  DLL_PUBLIC inline void Copy(const vector<Tensor<SrcBackend>> &other, cudaStream_t stream) {
    auto type = other[0].type();
    auto layout = other[0].GetLayout();

    vector<Dims> new_shape(other.size());
    for (size_t i = 0; i < other.size(); ++i) {
      assert(type == other[i].type());
      assert(layout == other[i].GetLayout());
      new_shape[i] = other[i].shape();
    }

    this->Resize(new_shape);
    if (IsValidType(type)) {
      this->set_type(type);
    }
    this->SetLayout(layout);

    for (size_t i = 0; i < other.size(); ++i) {
      type.template Copy<SrcBackend, Backend>(
          raw_mutable_tensor(i),
          other[i].raw_data(),
          other[i].size(), 0);
      this->meta_[i].SetSourceInfo(other[i].GetSourceInfo());
      this->meta_[i].SetSkipSample(other[i].ShouldSkipSample());
    }
  }

  /**
   * @brief Resize function to allocate a list of tensors. The input vector
   * contains a set of dimensions for each tensor to be allocated in the
   * list.
   */
  DLL_PUBLIC inline void Resize(const vector<Dims> &new_shape) {
    if (new_shape == shape_) return;

    // Calculate the new size
    Index num_tensor = new_shape.size(), new_size = 0;
    offsets_.resize(num_tensor);
    if (!new_shape.empty()) {
      size_t dim = new_shape[0].size();
      for (auto &s : new_shape) {
        DALI_ENFORCE(s.size() == dim, "All items should have same number of dimensions");
      }
    }
    for (Index i = 0; i < num_tensor; ++i) {
      auto tensor_size = volume(new_shape[i]);

      // Save the offset of the current sample & accumulate the size
      offsets_[i] = new_size;
      new_size += tensor_size;
    }
    DALI_ENFORCE(new_size >= 0, "Invalid negative buffer size.");

    // Resize the underlying allocation and save the new shape
    ResizeHelper(new_size);
    shape_ = new_shape;

    // Tensor view of this TensorList is no longer valid
    if (tensor_view_) {
      tensor_view_->ShareData(this);
    }

    meta_.resize(num_tensor, DALIMeta(layout_));
  }

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

    // Tensor view of this TensorList is no longer valid
    if (tensor_view_) {
      tensor_view_->ShareData(this);
    }

    // If the other tensor has a non-zero size allocation, mark that
    // we are now sharing an allocation with another buffer
    shares_data_ = num_bytes_ > 0 ? true : false;
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
  DLL_PUBLIC inline void ShareData(void *ptr, size_t bytes) {
    DALI_ENFORCE(ptr != nullptr, "Input pointer must not be nullptr.");

    // Save our new pointer and bytes. Reset our type, shape, and size
    data_.reset(ptr, [](void *) {});
    num_bytes_ = bytes;
    type_ = TypeInfo::Create<NoType>();
    shape_.clear();
    offsets_.clear();
    size_ = 0;

    // Tensor view of this TensorList is no longer valid
    if (tensor_view_) {
      tensor_view_->ShareData(this);
    }

    // If the input pointer stores a non-zero size allocation, mark
    // that we are sharing our underlying data
    shares_data_ = num_bytes_ > 0 ? true : false;
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
  inline const vector<Index> &tensor_shape(int idx) const {
#ifndef NDEBUG
    DALI_ENFORCE(idx >= 0, "Negative index not supported");
    DALI_ENFORCE((size_t)idx < shape_.size(), "Index out of offset range");
#endif
    return shape_[idx];
  }

  /**
   * @brief Returns the shape of the entire TensorList.
   */
  inline const vector<Dims> &shape() const {
    return shape_;
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
      offset += volume(o);
    }
    return true;
  }

  /**
   * @brief Returns the number of elements
   *  in the TensorList
   */
  inline size_t GetElementsNumber() const {
    size_t elms = 0;

    for (auto &shape : shape_) {
      elms += volume(shape);
    }
    return elms;
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


  // So we can access the members of other TensorListes
  // with different template types
  template <typename InBackend>
  friend class TensorList;

  DISABLE_COPY_MOVE_ASSIGN(TensorList);

  inline std::string GetSourceInfo(int idx) const {
    return meta_[idx].GetSourceInfo();
  }

  inline void SetSourceInfo(int idx, const std::string& source_info) {
    meta_[idx].SetSourceInfo(source_info);
  }

  inline DALITensorLayout GetLayout() const {
    // Layout is enforced to be the same across all the samples
    return layout_;
  }

  inline void SetLayout(DALITensorLayout layout) {
    // Layout is enforced to be the same across all the samples
    layout_ = layout;
    for (auto& meta : meta_)
      meta.SetLayout(layout_);
  }

  inline void SetSkipSample(int idx, bool skip_sample) {
    return meta_[idx].SetSkipSample(skip_sample);
  }

  inline bool ShouldSkipSample(int idx) const {
    return meta_[idx].ShouldSkipSample();
  }

 protected:
  // We store a set of dimension for each tensor in the list.
  // We also pre-compute the offsets of each tensor in the
  // underlying allocation for random access
  vector<Dims> shape_;
  vector<Index> offsets_;
  vector<DALIMeta> meta_;
  DALITensorLayout layout_;

  // In order to not leak memory (and make it slightly faster)
  // when sharing data with a Tensor, we will store a pointer to
  // Tensor that shares the data with this TensorList (valid only
  // if IsDenseTensor returns true)
  Tensor<Backend> * tensor_view_;

  USE_BUFFER_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_LIST_H_
