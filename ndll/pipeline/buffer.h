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
 * Uses input 'Backend' type to allocate and free memory.
 */
template <typename Backend>
class Buffer {
public:
  inline Buffer() : owned_(true), data_(nullptr), size_(0), true_size_(0) {}

  virtual ~Buffer() {
    backend_.Delete(data_, true_size_*type_.size());
  }

  /**
   * @brief Resizes the buffer to fit `size` elements. 
   * The underlying storage is only reallocated in the case that
   * the current buffer is not large enough for the requested 
   * number of elements.
   */
  inline virtual void Resize(int new_size) {
    if (new_size == size_) return;
    NDLL_ENFORCE(owned_, "Buffer does not own underlying "
        "storage, calling 'Resize()' not allowed");
    if (type_.id() == NO_TYPE) {
      // If the type has not been set yet, we just set the size
      // and shape of the buffer and do not allocate any memory.
      // Any previous resize dims are overwritten.
      size_ = new_size;
      true_size_ = new_size;
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

  inline Dim size() const { return size_; }
  
  inline size_t nbytes() const {
    return size_*type_.size();
  }

  inline TypeMeta type() const {
    return type_;
  }
  
  DISABLE_COPY_MOVE_ASSIGN(Buffer);
protected:
  Backend backend_;
  TypeMeta type_;

  // Indicates if this object owns the underlying data.
  // Buffers always own their underlying storage.
  bool owned_;

  // Pointer to underlying storage & meta-data
  void *data_;
  Dim size_;
  
  // To keep track of the true size
  // of the underlying allocation
  Dim true_size_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_BUFFER_H_
