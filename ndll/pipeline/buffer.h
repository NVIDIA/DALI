#ifndef NDLL_PIPELINE_BUFFER_H_
#define NDLL_PIPELINE_BUFFER_H_

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
 * @brief defines the non-templated interface for basic 
 * memory storage for ops
 */
class BufferBase {
public:
  BufferBase() : size_(0), true_size_(0) {}
  virtual ~BufferBase() = default;

  vector<Dim> shape() const {
    return shape_;
  }
  
  // Used by the pipeline to manage copies of the data
  virtual void* raw_data() = 0;
  virtual const void* raw_data() const = 0;
  virtual size_t bytes() const = 0;
  
  DISABLE_COPY_ASSIGN(BufferBase);
protected:
  int size_;
  vector<Dim> shape_;
  
  // To keep track of the true size
  // of the underlying allocation
  int true_size_;
};

/**
 * @brief the basic unit of data storage for the Pipeline and Operators.
 * Uses input 'Backend' type to allocate and free memory. Supports both 
 * dense and jagged tensors.
 */
template <typename Backend>
class Buffer : public BufferBase {
public:
  Buffer() : data_(nullptr) {}
  virtual ~Buffer() = default;

  /**
   * @brief Resizes the buffer to fit `size` elements. 
   * The underlying storage is only reallocated in the case that
   * the current buffer is not large enough for the requested 
   * number of elements.
   */
  void Resize(int size) {
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
  void Resize(vector<Dim> shape) {
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
      Backend::Delete(data_, true_size_*type_.size());
      Backend::New(&data_, new_size*type_.size());
      true_size_ = new_size;
    }

    // If we have enough storage already allocated, don't reallocate
    size_ = new_size;
    shape_ = shape;
  }

  template <typename T>
  T* data() {
    // If the buffer has no type, set the type to the the
    // calling type and allocate the buffer
    if (type_.id() == NO_TYPE) {
      type_.SetType<T>();

      // Allocate memory for the size
      NDLL_ENFORCE(data_ == nullptr,
          "data ptr is non-nullptr, something has gone wrong");

      // TODO(tgale): If true_size == 0, make sure this keeps our pointer
      // set to nullptr
      Backend::New(data_, true_size_*type_.size());
    }
    NDLL_ENFORCE(type_.id() == TypeTable::GetTypeID<T>(),
        "Calling type does not match buffer data type: " +
        TypeTable::GetTypeName<T>() + " v. " + type_.name());
    
    return data_;
  }

  template <typename T>
  const T* data() const {
    NDLL_ENFORCE(type_.id() != NO_TYPE,
        "Buffer has no type, 'data<T>()' must be called "
        "on non-const buffer to set valid type");
    NDLL_ENFORCE(type_.id() == TypeTable::GetTypeID<T>(),
        "Calling type does not match buffer data type: " +
        TypeTable::GetTypeName<T>() + " v. " + type_.name());
    return data_;
  }
  
  void* raw_data() override {
    NDLL_ENFORCE(type_.id() != NO_TYPE,
        "Buffer has no type, 'data<T>()' must be called "
        "on non-const buffer to set valid type");
    return static_cast<void*>(data_);
  }
  
  const void* raw_data() const override {
    NDLL_ENFORCE(type_.id() != NO_TYPE,
        "Buffer has no type, 'data<T>()' must be called "
        "on non-const buffer to set valid type");
    return static_cast<void*>(data_);
  }
  
  size_t bytes() const {
    return size_*type_.size();
  }

  DISABLE_COPY_ASSIGN(Buffer);
private:
  TypeMeta type_;
  void *data_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_BUFFER_H_
