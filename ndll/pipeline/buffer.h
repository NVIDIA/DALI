#ifndef NDLL_PIPELINE_BUFFER_H_
#define NDLL_PIPELINE_BUFFER_H_

#include <numeric>
#include <functional>

#include "ndll/common.h"
#include "ndll/error_handling.h"

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
  BufferBase() : size_(0), true_bytes_(0) {}
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
  size_t true_bytes_;
};

/**
 * @brief the basic unit of data storage for the Pipeline and Operators.
 * Uses input 'Backend' type to allocate and free memory. Supports both 
 * dense and jagged tensors.
 */
template <typename Backend, typename T>
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
    size_t new_bytes = new_size*sizeof(T);
    NDLL_ENFORCE(new_size > 0);

    if (new_bytes > true_bytes_) {
      Backend::Delete(data_, true_bytes_);
      Backend::New(data_, new_bytes);
      true_bytes_ = new_bytes;
    }
    // Store the new buffer size
    size_ = new_size;
    shape_ = shape;
  }


  T* data() { return data_; }

  const T* data() const { return data_; }

  vector<Dim> shape() const override {
    return shape_;
  }
  
  void* raw_data() override {
    return static_cast<void*>(data_);
  }
  
  const void* raw_data() const override {
    return static_cast<void*>(data_);
  }
  
  size_t bytes() const {
    return size_*sizeof(T);
  }

  DISABLE_COPY_ASSIGN(Buffer);
private:
  T *data_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_BUFFER_H_
