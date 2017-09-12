#ifndef NDLL_PIPELINE_BUFFER_H_
#define NDLL_PIPELINE_BUFFER_H_

#include <limits>
#include <numeric>
#include <functional>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/types.h"

namespace ndll {

// Basic type for Buffer dimensions
typedef int64_t Dim;

// Helper function to get product of dims
inline Dim Product(const vector<Dim> &shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<Dim>());
}

/**
 * @brief the basic unit of data storage for the Pipeline and Operators.
 * Uses input 'Backend' type to allocate and free memory. Supports both 
 * dense and jagged tensors.
 */
template <typename Backend>
class Buffer {
public:
  inline Buffer() : owned_(true), data_(nullptr), size_(0), true_size_(0) {}
  virtual ~Buffer() = default;

  /**
   * @brief Resizes the buffer to fit `size` elements. 
   * The underlying storage is only reallocated in the case that
   * the current buffer is not large enough for the requested 
   * number of elements.
   */
  inline void Resize(int size) {
    Resize(vector<Dim>{size});
  }

  /**
   * @brief Resizes the buffer to fit `Product(shape)` elements.
   * The underlying storage is only reallocated in the case that
   * the current buffer is not large enough for the requested 
   * number of elements.
   *
   * Passing in a shape of '{}' indicates a scalar value
   */
  inline void Resize(vector<Dim> shape) {
    NDLL_ENFORCE(owned_, "Buffer does not own underlying "
        "storage, calling 'Resize()' not allowed");
    int new_size = Product(shape);
    if (type_.id() == NO_TYPE) {
      // If the type has not been set yet, we just set the size
      // and shape of the buffer and do not allocate any memory.
      // Any previous resize dims are overwritten.
      size_ = new_size;
      true_size_ = new_size;
      shape_ = shape;
      return;
    }

    if (new_size > true_size_) {
      // Re-allocate the buffer to meet the new size requirements
      backend_.Delete(data_, true_size_*type_.size());
      data_ = backend_.New(new_size*type_.size());
      true_size_ = new_size;
    }

    // If we have enough storage already allocated, don't reallocate
    size_ = new_size;
    shape_ = shape;
  }

  template <typename T>
  inline T* data() {
    // If the buffer has no type, set the type to the the
    // calling type and allocate the buffer
    if (type_.id() == NO_TYPE) {
      NDLL_ENFORCE(data_ == nullptr,
          "data ptr is non-nullptr, something has gone wrong");
      NDLL_ENFORCE(owned_,
          "Buffer does not have type and does not own underlying "
          "storage. Calling 'data' not allowed");
      type_.SetType<T>();
      
      // TODO(tgale): If true_size == 0, make sure this
      // keeps our pointer set to nullptr
      data_ = backend_.New(true_size_*type_.size());
    }
    NDLL_ENFORCE(type_.id() == TypeTable::GetTypeID<T>(),
        "Calling type does not match buffer data type: " +
        TypeTable::GetTypeName<T>() + " v. " + type_.name());
    
    return data_;
  }

  template <typename T>
  inline const T* data() const {
    NDLL_ENFORCE(type_.id() != NO_TYPE,
        "Buffer has no type, 'data<T>()' must be called "
        "on non-const buffer to set valid type");
    NDLL_ENFORCE(type_.id() == TypeTable::GetTypeID<T>(),
        "Calling type does not match buffer data type: " +
        TypeTable::GetTypeName<T>() + " v. " + type_.name());
    return data_;
  }
  
  inline void* raw_data() {
    NDLL_ENFORCE(type_.id() != NO_TYPE,
        "Buffer has no type, 'data<T>()' must be called "
        "on non-const buffer to set valid type");
    return static_cast<void*>(data_);
  }
  
  inline const void* raw_data() const {
    NDLL_ENFORCE(type_.id() != NO_TYPE,
        "Buffer has no type, 'data<T>()' must be called "
        "on non-const buffer to set valid type");
    return static_cast<void*>(data_);
  }

  inline int ndim() const {
    return shape_.size();
  }

  inline Dim dim(int idx) const {
#ifdef DEBUG
    NDLL_ENFORCE(i < shape_.size(), "index exceeds ndim");
    NDLL_ENFORCE(i > 0, "negative index not supported");
#endif
    return shape_[idx];
  }

  inline int dimi(int idx) const {
#ifdef DEBUG
    NDLL_ENFORCE(i < shape_.size(), "index exceeds ndim");
    NDLL_ENFORCE(i > 0, "negative index not supported");
#endif
    NDLL_ENFORCE(shape_[idx] < std::numeric_limits<int>::max());
    return shape_[idx];
  }

  inline Dim size() const { return size_; }
  
  inline vector<Dim> shape() const {
    return shape_;
  }
  
  inline size_t nbytes() const {
    return size_*type_.size();
  }

  inline TypeMeta type() const {
    return type_;
  }
  
  DISABLE_COPY_ASSIGN(Buffer);
protected:
  Backend backend_;
  TypeMeta type_;

  // Indicates if this object owns the underlying data
  bool owned_;

  // Pointer to underlying storage & meta-data
  void *data_;
  Dim size_;
  vector<Dim> shape_;
  
  // To keep track of the true size
  // of the underlying allocation
  Dim true_size_;
};

/**
 * Wraps a portion of a buffer. Underlying memory is not 
 * owned by the buffer and cannot be resized.
 */
template <typename Backend>
class SubBuffer : public Buffer<Backend> {
public:
  /**
   * @brief Construct a sub-buffer that wraps a single datum from 
   * the input buffer. Outer dimension of the buffer is assumed 
   * to be the samples dimension i.e. 'N'
   */
  inline SubBuffer(Buffer<Backend> *buffer, int data_idx) {
    Reset(buffer, data_idx);
  }

  inline void Reset(Buffer<Backend> *buffer, int data_idx) {
    // We require a sample dimension and that
    // the data_idx is in the valid range
    NDLL_ENFORCE(buffer->ndim() > 0);
    NDLL_ENFORCE(data_idx < buffer->shape()[0] && data_idx >= 0);

    // The sub-buffer does not own its memory
    this->owned_ = false;
    
    // TODO(tgale): We need to handle jagged tensors. We need to
    // grab the correct dims for the image & get the offset for
    // the pointer. To make this simple we could create a subclass
    // of the Buffer for batches w/ some nice utilities that we
    // want to operate on batches of data, then just query this
    // for the meta-data we need
    
    // Get dimensions of the datum
    this->shape_.insert(
        this->shape_.begin(),
        buffer->shape().begin() + 1,
        buffer->shape().end());
    this->true_size_ = buffer->size() / buffer->dim(0);
    this->size_ = this->true_size_;

    // Save the offset pointer & type info
    this->type_ = buffer->type();
    this->data_ = nullptr;
    int data_offset = data_idx * this->true_size_ * this->type_.size();
    if (buffer->raw_data() != nullptr) {
      // Offset the pointer in bytes
      this->data_ = static_cast<void*>(
          static_cast<uint8*>(buffer->raw_data()) + data_offset);
    }
  }
  
protected:
};

} // namespace ndll

#endif // NDLL_PIPELINE_BUFFER_H_
