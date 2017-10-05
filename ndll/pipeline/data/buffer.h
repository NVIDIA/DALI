#ifndef NDLL_PIPELINE_DATA_BUFFER_H_
#define NDLL_PIPELINE_DATA_BUFFER_H_

#include <limits>
#include <numeric>
#include <functional>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/types.h"

namespace ndll {

// Basic data type for our indices and dimension sizes
typedef int64_t Index;

// Helper function to get product of dims
inline Index Product(const vector<Index> &shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<Index>());
}

// Helper function to get a string of the data shape
inline string ShapeString(vector<Index> shape) {
  string tmp;
  for (auto &val : shape) tmp += std::to_string(val) + " ";
  return tmp;
}

/**
 * @brief Base class to provide common functionality needed by Pipeline data
 * structures. Not meant for use, does not provide methods for allocating
 * any actual storage. The 'Backend' template parameter dictates where the
 * underlying storage is located (CPU or GPU).
 *
 * Data storage types in NDLL use delayed allocation, and have a small 
 * custom type system that allows us to circumvent template paramters.
 * This is turn allows the Pipeline to manage all intermediate memory,
 * opening the door for optimizations and reducing the work that must
 * be done by the user when defining operations.
 */
template <typename Backend>
class Buffer {
public:
  /**
   * @brief Initializes a buffer of size 0.
   */
  inline Buffer() : data_(nullptr), size_(0), true_size_(0) {}

  /**
   * @brief Cleans up underlying storage.
   */
  virtual ~Buffer() {
    if (true_size_*type_.size() > 0) {
      backend_.Delete(data_, true_size_*type_.size());
    }
  }

  /**
   * @brief Returns a typed pointer to the underlying storage. If the
   * buffer has not been allocated because it does not yet have a type,
   * the calling type is taken to be the type of the data. The memory
   * is allocated, and the type remains fixed for the lifetime of the
   * object.
   *
   * If the buffer already has a valid type, and the calling type does
   * not match, a std::runtime_error is thrown.
   */
  template <typename T>
  inline T* data() {
    // If the buffer has no type, set the type to the the
    // calling type and allocate the buffer
    if (type_.id() == NO_TYPE) {
      NDLL_ENFORCE(data_ == nullptr,
          "data ptr is non-nullptr, something has gone wrong");

      type_.SetType<T>();
      NDLL_ENFORCE(type_.size() > 0,
          "Set datatype must have non-zero element size");
      
      // Make sure we keep our nullptr if we don't
      // have anything to allocate
      if (true_size_ > 0) {
        data_ = backend_.New(true_size_*type_.size());
      }
    }
    NDLL_ENFORCE(type_.id() == TypeTable::GetTypeID<T>(),
        "Calling type does not match buffer data type: " +
        TypeTable::GetTypeName<T>() + " v. " + type_.name());
    
    return static_cast<T*>(data_);
  }

  /**
   * @brief Returns a const, typed pointer to the underlying storage.
   * A valid type must be set prior to calling this method by calling
   * the non-const version of the method, or calling 'set_type'.
   */
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

  /**
   * @brief Return an un-typed pointer to the underlying storage.
   * A valid type must be set prior to calling this method by calling
   * the non-const version of the method, or calling 'set_type'.
   */
  inline void* raw_data() {
    NDLL_ENFORCE(type_.id() != NO_TYPE,
        "Buffer has no type, 'data<T>()' must be called "
        "on non-const buffer to set valid type");
    return static_cast<void*>(data_);
  }

  /**
   * @brief Return an const, un-typed pointer to the underlying storage.
   * A valid type must be set prior to calling this method by calling
   * the non-const version of the method, or calling 'set_type'.
   */
  inline const void* raw_data() const {
    NDLL_ENFORCE(type_.id() != NO_TYPE,
        "Buffer has no type, 'data<T>()' must be called "
        "on non-const buffer to set valid type");
    return static_cast<void*>(data_);
  }

  /**
   * @brief Returns the size in elements of the underlying data
   */
  inline Index size() const { return size_; }

  /**
   * @brief Returns the size in bytes of the underlying data
   */
  inline size_t nbytes() const {
    return size_*type_.size();
  }

  /**
   * @brief Returns the TypeMeta object that keeps track of the 
   * datatype of the underlying storage.
   */
  inline TypeMeta type() const {
    return type_;
  }

  /**
   * @brief Sets the type of the buffer. Throws an error if the underlying
   * storage already has a type. If the buffer has no type but has non-zero 
   * size, we allocate the memory.
   */
  inline void set_type(TypeMeta type) {
    if (type.id() == type_.id()) return;
    NDLL_ENFORCE(type_.id() == NO_TYPE, "Buffer already has valid type");
    NDLL_ENFORCE(type.size() > 0,
        "Set datatype must have non-zero element size");
    NDLL_ENFORCE(data_ == nullptr,
        "Something has gone wrong. Untyped buffer cannot store data");
    type_ = type;

    // If the buffer has a set size allocate the
    // memory for the size of the buffer
    if (true_size_ > 0) {
      data_ = backend_.New(true_size_*type_.size());
    }
  }
  
  DISABLE_COPY_MOVE_ASSIGN(Buffer);
protected:
  Backend backend_;
  TypeMeta type_;

  // Pointer to underlying storage & meta-data
  void *data_;
  Index size_;
  
  // To keep track of the true size
  // of the underlying allocation
  Index true_size_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_DATA_BUFFER_H_
