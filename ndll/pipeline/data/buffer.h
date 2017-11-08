#ifndef NDLL_PIPELINE_DATA_BUFFER_H_
#define NDLL_PIPELINE_DATA_BUFFER_H_

#include <limits>
#include <numeric>
#include <functional>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/types.h"

namespace ndll {

// Helper function to get product of dims
inline Index Product(const vector<Index> &shape) {
  if (shape.size() == 0) return 0;
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<Index>());
}

// Helper function to get a string of the data shape
inline string ShapeString(vector<Index> shape) {
  string tmp;
  for (auto &val : shape) tmp += std::to_string(val) + " ";
  return tmp;
}

// NOTE: Data storage types in NDLL use delayed allocation, and have a
// small custom type system that allows us to circumvent template
// paramters. This is turn allows the Pipeline to manage all intermediate
// memory, opening the door for optimizations and reducing the work that
// must be done by the user when defining operations.
  
/**
 * @brief Base class to provide common functionality needed by Pipeline data
 * structures. Not meant for use, does not provide methods for allocating
 * any actual storage. The 'Backend' template parameter dictates where the
 * underlying storage is located (CPU or GPU).
 *
 * Buffers are untyped on construction, and don't receive a valid type until
 * 'set_type' or 'data<T>()' is called on a non-const buffer. Upon receiving
 * a valid type, the underlying storage for the buffer is allocated. The type
 * of the underlying data can change over the lifetime of an object if 
 * 'set_type' or 'data<T>()' is called again where the calling type does not
 * match the underlying type on the buffer. In this case, the Buffer swaps its
 * current type, but only re-allocates memory if it does not have enough bytes
 * of allocated storage to store the number of elements in the buffer with the
 * new data type size.
 */
template <typename Backend>
class Buffer {
public:
  /**
   * @brief Initializes a buffer of size 0.
   */
  inline Buffer() : data_(nullptr), size_(0), shares_data_(false), num_bytes_(0) {}

  virtual ~Buffer() = default;

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
  inline T* mutable_data() {
    // Note: Call to 'set_type' will immediately return if the calling
    // type matches the current type of the buffer.
    TypeInfo calling_type;
    calling_type.SetType<T>();
    set_type(calling_type);
    return static_cast<T*>(data_.get());
  }

  /**
   * @brief Returns a const, typed pointer to the underlying storage.
   * The calling type must match the underlying type of the buffer.
   */
  template <typename T>
  inline const T* data() const {
    NDLL_ENFORCE(IsValidType(type_),
        "Buffer has no type, 'mutable_data<T>()' must be called "
        "on non-const buffer to set valid type");
    NDLL_ENFORCE(type_.id() == TypeTable::GetTypeID<T>(),
        "Calling type does not match buffer data type: " +
        TypeTable::GetTypeName<T>() + " v. " + type_.name());
    return static_cast<T*>(data_.get());
  }

  /**
   * @brief Return an un-typed pointer to the underlying storage.
   * A valid type must be set prior to calling this method by calling
   * the non-const version of the method, or calling 'set_type'.
   */
  inline void* raw_mutable_data() {
    NDLL_ENFORCE(IsValidType(type_),
        "Buffer has no type, 'mutable_data<T>()' or 'set_type' must "
        "be called on non-const buffer to set valid type");
    return static_cast<void*>(data_.get());
  }

  /**
   * @brief Return an const, un-typed pointer to the underlying storage.
   * A valid type must be set prior to calling this method by calling
   * the non-const version of the method, or calling 'set_type'.
   */
  inline const void* raw_data() const {
    NDLL_ENFORCE(IsValidType(type_),
        "Buffer has no type, 'mutable_data<T>()' or 'set_type' must "
        "be called on non-const buffer to set valid type");
    return static_cast<void*>(data_.get());
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
   * @brief Returns the TypeInfo object that keeps track of the 
   * datatype of the underlying storage.
   */
  inline TypeInfo type() const {
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
  inline void set_type(TypeInfo new_type) {
    if (new_type.id() == type_.id()) return;
    NDLL_ENFORCE(new_type.size() > 0,
        "New datatype must have non-zero element size");

    if (!IsValidType(type_)) {
      // If the buffer has no type, set the type to the
      // calling type and allocate the buffer
      NDLL_ENFORCE(data_ == nullptr,
          "Data ptr is nullptr, something has gone wrong.");
      type_ = new_type;

      // Make sure we keep our nullptr if we don't
      // have anything to allocate
      num_bytes_ = size_ * type_.size();
      if (num_bytes_ > 0) {
        data_.reset(Backend::New(num_bytes_), std::bind(
                &Buffer<Backend>::DeleterHelper,
                this, std::placeholders::_1,
                type_, size_));
        type_.template Construct<Backend>(data_.get(), size_);
      }
    } else {
      // If the calling type does not match the current buffer
      // type, reset the type and re-allocate the memory if
      // we do not have enough
      size_t new_num_bytes = size_ * new_type.size();
      if (new_num_bytes > num_bytes_) {
        // Re-allocate the underlying storage
        data_.reset(Backend::New(new_num_bytes), std::bind(
                &Buffer<Backend>::DeleterHelper,
                this, std::placeholders::_1,
                type_, size_));
        num_bytes_ = new_num_bytes;
        shares_data_ = false;
      }

      // Save the new type
      type_ = new_type;
      type_.template Construct<Backend>(data_.get(), size_);
    }
  }

  // Helper function for cleaning up data storage. This unfortunately
  // has to be public so that we can bind it into the deleter of our
  // shared pointers
  void DeleterHelper(void *ptr, TypeInfo type, Index size) {
    type.template Destruct<Backend>(ptr, size);
    Backend::Delete(ptr, size*type.size());
  }
  
  DISABLE_COPY_MOVE_ASSIGN(Buffer);
protected:
  // Helper to resize the underlying allocation
  inline void ResizeHelper(Index new_size) {
    NDLL_ENFORCE(new_size >= 0, "Input size less than zero not supported.");
    
    if (!IsValidType(type_)) {
      // If the type has not been set yet, we just set the size of the
      // buffer and do not allocate any memory. Any previous size is
      // overwritten.
      NDLL_ENFORCE(data_ == nullptr, "Buffer has no type, data_ should be nullptr.");
      NDLL_ENFORCE(num_bytes_ == 0, "Buffer has no type, num_bytes_ should be 0.");
      
      size_ = new_size;
      return;
    }

    size_t new_num_bytes = new_size * type_.size();
    if (new_num_bytes > num_bytes_) {
      data_.reset(Backend::New(new_num_bytes), std::bind(
              &Buffer<Backend>::DeleterHelper,
              this, std::placeholders::_1,
              type_, new_size));
      num_bytes_ = new_num_bytes;
      
      // Call the constructor for the underlying datatype
      type_.template Construct<Backend>(data_.get(), new_size);
      
      // If we were sharing data, we aren't anymore
      shares_data_ = false;
    }

    size_ = new_size;
  }
  
  Backend backend_;
  
  TypeInfo type_; // Data type of underlying storage
  shared_ptr<void> data_; // Pointer to underlying storage
  Index size_; // The number of elements in the buffer
  bool shares_data_;
  
  // To keep track of the true size
  // of the underlying allocation
  size_t num_bytes_;
};

// Macro so we don't have to list these in all
// classes that derive from Buffer
#define USE_BUFFER_MEMBERS()                    \
  using Buffer<Backend>::ResizeHelper;          \
  using Buffer<Backend>::backend_;              \
  using Buffer<Backend>::type_;                 \
  using Buffer<Backend>::data_;                 \
  using Buffer<Backend>::size_;                 \
  using Buffer<Backend>::shares_data_;          \
  using Buffer<Backend>::num_bytes_

} // namespace ndll

#endif // NDLL_PIPELINE_DATA_BUFFER_H_
