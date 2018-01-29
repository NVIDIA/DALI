// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_DATA_TENSOR_LIST_H_
#define NDLL_PIPELINE_DATA_TENSOR_LIST_H_

#include <cstring>
#include <vector>

#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/buffer.h"

namespace ndll {

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
class TensorList : public Buffer<Backend> {
 public:
  TensorList() {}
  ~TensorList() = default;

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
  inline void Copy(const TensorList<SrcBackend> &other, cudaStream_t stream) {
    this->set_type(other.type());
    ResizeLike(other);
    type_.template Copy<Backend, SrcBackend>(this->raw_mutable_data(),
        other.raw_data(), this->size(), stream);
  }

  /**
   * @brief Resize function to allocate a list of tensors. The input vector
   * contains a set of dimensions for each tensor to be allocated in the
   * list.
   */
  inline void Resize(const vector<Dims> &new_shape) {
    if (new_shape == shape_) return;

    // Calculate the new size
    Index num_tensor = new_shape.size(), new_size = 0;
    offsets_.resize(num_tensor);
    for (Index i = 0; i < num_tensor; ++i) {
      auto tensor_size = Product(new_shape[i]);

      // Save the offset of the current sample & accumulate the size
      offsets_[i] = new_size;
      new_size += tensor_size;
    }
    NDLL_ENFORCE(new_size >= 0, "Invalid negative buffer size.");

    // Resize the underlying allocation and save the new shape
    ResizeHelper(new_size);
    shape_ = new_shape;
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
  inline void ShareData(TensorList<Backend> *other) {
    NDLL_ENFORCE(other != nullptr, "Input TensorList is nullptr");
    NDLL_ENFORCE(IsValidType(other->type_), "To share data, "
        "the input TensorList must have a valid data type");

    // Save the calling TensorLists meta-data
    data_ = other->data_;
    shape_ = other->shape_;
    size_ = other->size_;
    offsets_ = other->offsets_;
    type_ = other->type_;
    num_bytes_ = other->num_bytes_;

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
  inline void ShareData(void *ptr, size_t bytes) {
    NDLL_ENFORCE(ptr != nullptr, "Input pointer must not be nullptr.");

    // Save our new pointer and bytes. Reset our type, shape, and size
    data_.reset(ptr, [](void *) {});
    num_bytes_ = bytes;
    type_ = TypeInfo::Create<NoType>();
    shape_.clear();
    offsets_.clear();
    size_ = 0;

    // If the input pointer stores a non-zero size allocation, mark
    // that we are sharing our underlying data
    shares_data_ = num_bytes_ > 0 ? true : false;
  }

  /**
   * @brief Returns a typed pointer to the tensor with the given index.
   */
  template <typename T>
  inline T* mutable_tensor(int idx) {
    return this->template mutable_data<T>() + tensor_offset(idx);
  }

  /**
   * @brief Returns a const typed pointer to the tensor with the given index.
   */
  template <typename T>
  inline const T* tensor(int idx) const {
    return this->template data<T>() + tensor_offset(idx);
  }

  /**
   * @brief Returns a raw pointer to the tensor with the given index.
   */
  inline void* raw_mutable_tensor(int idx) {
    return static_cast<void*>(
        static_cast<uint8*>(this->raw_mutable_data()) +
        (tensor_offset(idx) * type_.size()));
  }

  /**
   * @brief Returns a const raw pointer to the tensor with the given index.
   */
  inline const void* raw_tensor(int idx) const {
    return static_cast<const void*>(
        static_cast<const uint8*>(this->raw_data()) +
        (tensor_offset(idx) * type_.size()));
  }

  /**
   * @brief Returns the number of tensors in the list.
   */
  inline int ntensor() const {
    return shape_.size();
  }

  /**
   * @brief Returns the offset of the tensor with the given index.
   */
  inline Index tensor_offset(int idx) const {
#ifndef NDEBUG
    NDLL_ENFORCE(idx >= 0, "Negative index not supported");
    NDLL_ENFORCE((size_t)idx < offsets_.size(), "Index out of offset range");
#endif
    return offsets_[idx];
  }

  /**
   * @brief Return the shape of the tensor with the given index.
   */
  inline vector<Index> tensor_shape(int idx) const {
#ifndef NDEBUG
    NDLL_ENFORCE(idx >= 0, "Negative index not supported");
    NDLL_ENFORCE((size_t)idx < shape_.size(), "Index out of offset range");
#endif
    return shape_[idx];
  }

  /**
   * @brief Returns the shape of the entire TensorList.
   */
  inline vector<Dims> shape() const {
    return shape_;
  }

  /**
   * @brief Checks whether all of the tensors
   * stored in TensorList have the same shape
   */
  inline bool IsTensor() const {
    if (ntensor() == 0) {
      return true;
    }
    const Dims& d = shape_[0];
    for (const auto& o : shape_) {
      if (d != o) {
        return false;
      }
    }
    return true;
  }

  // So we can access the members of other TensorListes
  // with different template types
  template <typename InBackend>
  friend class TensorList;

  DISABLE_COPY_MOVE_ASSIGN(TensorList);

 protected:
  // We store a set of dimension for each tensor in the list.
  // We also pre-compute the offsets of each tensor in the
  // underlying allocation for random access
  vector<Dims> shape_;
  vector<Index> offsets_;

  USE_BUFFER_MEMBERS();
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_DATA_TENSOR_LIST_H_
