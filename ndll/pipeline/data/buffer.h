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
  inline Buffer() : data_(nullptr), size_(0), num_bytes_(0) {}

  /**
   * @brief Cleans up underlying storage.
   */
  virtual ~Buffer() {
    if (num_bytes_*type_.size() > 0) {
      Backend::Delete(data_, num_bytes_*type_.size());
    }
  }

  /**
   * @brief Returns a typed pointer to the underlying storage. If the
   * buffer has not been allocated because it does not yet have a type,
   * the calling type is taken to be the type of the data and the memory
   * is allocated.
   *
   * If the buffer already has a valid type, and the calling type does
   * not match, the type of the buffer is reset and the underlying
   * storage is re-allocated if the buffer does not currently own
   * enough memory to store the current number of elements with the 
   * new data type.
   */
  template <typename T>
  inline T* data() {
    // Note: Call to 'set_type' will immediately return if the calling
    // type matches the current type of the buffer.
    TypeMeta calling_type;
    calling_type.SetType<T>();
    set_type(calling_type);
    return static_cast<T*>(data_);
  }

  /**
   * @brief Returns a const, typed pointer to the underlying storage.
   * The calling type must match the underlying type of the buffer.
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
        "Buffer has no type, 'data<T>()' or 'set_type' must "
        "be called on non-const buffer to set valid type");
    return static_cast<void*>(data_);
  }

  /**
   * @brief Return an const, un-typed pointer to the underlying storage.
   * A valid type must be set prior to calling this method by calling
   * the non-const version of the method, or calling 'set_type'.
   */
  inline const void* raw_data() const {
    NDLL_ENFORCE(type_.id() != NO_TYPE,
        "Buffer has no type, 'data<T>()' or 'set_type' must "
        "be called on non-const buffer to set valid type");
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
    // Note: This returns the number of bytes occupied by the current
    // number of elements stored in the buffer. This is not neccessarily
    // the number of bytes of the underlying allocation (num_bytes_)
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
   * @brief Sets the type of the buffer. If the buffer has not been 
   * allocated because it does not yet have a type, the calling type 
   * is taken to be the type of the data and the memory is allocated.
   *
   * If the buffer already has a valid type, and the calling type does
   * not match, the type of the buffer is reset and the underlying
   * storage is re-allocated if the buffer does not currently own
   * enough memory to store the current number of elements with the 
   * new data type.
   */
  inline void set_type(TypeMeta new_type) {
    if (new_type.id() == type_.id()) return;
    NDLL_ENFORCE(new_type.size() > 0,
        "New datatype must have non-zero element size");

    if (type_.id() == NO_TYPE) {
      // If the buffer has no type, set the type to the
      // calling type and allocate the buffer
      NDLL_ENFORCE("Data ptr is nullptr, something has gone wrong.");
      type_ = new_type;

      // Make sure we keep our nullptr if we don't
      // have anything to allocate
      num_bytes_ = size_ * type_.size();
      if (num_bytes_ > 0) {
        data_ = Backend::New(num_bytes_);
      }
    } else {
      // If the calling type does not match the current buffer
      // type, reset the type and re-allocate the memory if
      // we do not have enough
      size_t new_num_bytes = size_ * new_type.size();
      if (new_num_bytes > num_bytes_) {
        // Re-allocate the underlying storage
        Backend::Delete(data_, num_bytes_);
        data_ = Backend::New(new_num_bytes);
        num_bytes_ = new_num_bytes;
      }

      // Save the new type
      type_ = new_type;
    }
  }
  
  DISABLE_COPY_MOVE_ASSIGN(Buffer);
protected:
  Backend backend_;
  
  TypeMeta type_; // Data type of underlying storage
  void *data_; // Pointer to underlying storage
  Index size_; // The number of elements in the buffer
  
  // To keep track of the true size
  // of the underlying allocation
  size_t num_bytes_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_DATA_BUFFER_H_
