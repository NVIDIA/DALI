#ifndef NDLL_PIPELINE_DATA_TENSOR_H_
#define NDLL_PIPELINE_DATA_TENSOR_H_

#include <cstring>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/buffer.h"

namespace ndll {

/**
 * @brief Stores dense, multi-dimensional data. Provides utilities 
 * methods for handling dimensions and shapes of the stored data.
 *
 * Batch objects conform to the type management system defined in 
 * @ref Buffer.
 */
template <typename Backend>
class Tensor : public Buffer<Backend> {
public:
  Tensor() {}
  ~Tensor() = default;

  /**
   * Loads the tensor with data from the input vector
   */
  template <typename T>
  void Copy(const vector<T> &data, cudaStream_t stream = 0) {
    CopyHelper(this, data, stream);
  }
  
  /**
   * @brief Resizes the buffer to fit `Product(shape)` elements.
   * The underlying storage is only reallocated in the case that
   * the current buffer is not large enough for the requested 
   * number of elements.
   *
   * Passing in a shape of '{}' indicates a scalar value
   */
  inline virtual void Resize(const vector<Index> &shape) {
    Index new_size = Product(shape);
    if (type_.id() == NO_TYPE) {
      // If the type has not been set yet, we just set the size
      // and shape of the buffer and do not allocate any memory.
      // Any previous resize dims are overwritten.
      size_ = new_size;
      shape_ = shape;
      return;
    }

    size_t new_num_bytes = new_size*type_.size();
    if (new_num_bytes > num_bytes_) {      
      // Re-allocate the buffer to meet the new size requirements
      data_.reset(Backend::New(new_num_bytes),
          std::bind(
              &Backend::Delete,
              std::placeholders::_1,
              new_num_bytes)
          );
      num_bytes_ = new_num_bytes;
    }

    // If we have enough storage already allocated, don't re-allocate
    size_ = new_size;
    shape_ = shape;
  }

  /**
   * @brief Returns the shape of the Tensor
   */
  inline vector<Index> shape() const {
    return shape_;
  }

  /**
   * @brief Returns the number of dimensions of the Tensor
   */
  inline virtual int ndim() const {
    return shape_.size();
  }

  /**
   * @brief Returns the size of the dimension at the given index
   */
  inline virtual Index dim(int idx) const {
#ifndef NDEBUG
    NDLL_ENFORCE((size_t)idx < shape_.size(), "index exceeds ndim");
    NDLL_ENFORCE(idx >= 0, "negative index not supported");
#endif
    return shape_[idx];
  }
  
  DISABLE_COPY_MOVE_ASSIGN(Tensor);
protected:
  vector<Index> shape_;

  // So we don't have to put 'this->' everywhere
  using Buffer<Backend>::backend_;
  using Buffer<Backend>::type_;
  using Buffer<Backend>::data_;
  using Buffer<Backend>::size_;
  using Buffer<Backend>::num_bytes_;
};

// Note: CopyHelper lets us specialize on the Tensor backend type without
// specializing on the input vectors data type.
template <typename T>
void CopyHelper(Tensor<CPUBackend> *tensor, const vector<T> &data, cudaStream_t stream) {
  tensor->template mutable_data<T>();
  tensor->Resize({(Index)data.size()});
  std::memcpy(tensor->raw_mutable_data(), data.data(), tensor->nbytes(), stream);
}

template <typename T>
void CopyHelper(Tensor<GPUBackend> *tensor, const vector<T> &data, cudaStream_t stream) {
  tensor->template mutable_data<T>();
  tensor->Resize({(Index)data.size()});
  MemCopy(tensor->raw_mutable_data(), data.data(), tensor->nbytes(), stream);
}

} // namespace ndll

#endif // NDLL_PIPELINE_DATA_TENSOR_H_
