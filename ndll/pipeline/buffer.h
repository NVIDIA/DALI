#ifndef NDLL_PIPELINE_BUFFER_H_
#define NDLL_PIPELINE_BUFFER_H_

#include <limits>
#include <numeric>
#include <functional>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/types.h"

namespace ndll {

// TODO(tgale): Make sure the data storage hierarchy makes sense in terms of
// how the owned flag is used, where and when allocation occur, and when types
// are required to be setup

// Basic data type for our indices and dimension sizes
typedef int64_t Index;

// Helper function to get product of dims
inline Index Product(const vector<Index> &shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<Index>());
}

/**
 * @brief Base class for pipeline data storage classes. This should not be used,
 * it does not provide any method for altering the size of the underlying data
 */
template <typename Backend>
class Buffer {
public:
  inline Buffer() : owned_(true), data_(nullptr), size_(0), true_size_(0) {}

  virtual ~Buffer() {
    if (owned_) {
      backend_.Delete(data_, true_size_*type_.size());
    }
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
      
      // Make sure we keep our nullptr if we don't allocate anything
      size_t mem_size = true_size_*type_.size();
      if (mem_size != 0) {
        data_ = backend_.New(true_size_*type_.size());
      }
    }
    NDLL_ENFORCE(type_.id() == TypeTable::GetTypeID<T>(),
        "Calling type does not match buffer data type: " +
        TypeTable::GetTypeName<T>() + " v. " + type_.name());
    
    return static_cast<T*>(data_);
  }

  template <typename T>
  inline const T* data() const {
    NDLL_ENFORCE(type_.id() != NO_TYPE,
        "Buffer has no type, 'data<T>()' must be called "
        "on non-const buffer to set valid type");
    NDLL_ENFORCE(type_.id() == TypeTable::GetTypeID<T>(),
        "Calling type does not match buffer data type: " +
        TypeTable::GetTypeName<T>() + " v. " + type_.name());
    return static_cast<T*>(data_);
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

  inline Index size() const { return size_; }
  
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
  Index size_;
  
  // To keep track of the true size
  // of the underlying allocation
  Index true_size_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_BUFFER_H_
