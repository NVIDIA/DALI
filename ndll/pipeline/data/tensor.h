#ifndef NDLL_PIPELINE_DATA_TENSOR_H_
#define NDLL_PIPELINE_DATA_TENSOR_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/buffer.h"

namespace ndll {

/**
 * @brief Stores dense, multi-dimensional data. Provides utilities methods
 * for handling dimensions and shapes of the stored data.
 */
template <typename Backend>
class Tensor : public Buffer<Backend> {
public:
  Tensor() {}
  ~Tensor() = default;

  /**
   * @brief Resizes the buffer to fit `Product(shape)` elements.
   * The underlying storage is only reallocated in the case that
   * the current buffer is not large enough for the requested 
   * number of elements.
   *
   * Passing in a shape of '{}' indicates a scalar value
   */
  inline virtual void Resize(const vector<Index> &shape) {
    NDLL_ENFORCE(owned_, "Buffer does not own underlying "
        "storage, calling 'Resize()' not allowed");
    Index new_size = Product(shape);
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
    shape_ = shape;
  }

  inline vector<Index> shape() const {
    return shape_;
  }

  inline virtual int ndim() const {
    return shape_.size();
  }

  inline virtual Index dim(int idx) const {
#ifdef DEBUG
    NDLL_ENFORCE((size_t)idx < shape_.size(), "index exceeds ndim");
    NDLL_ENFORCE((size_t)idx >= 0, "negative index not supported");
#endif
    return shape_[idx];
  }
  
  DISABLE_COPY_MOVE_ASSIGN(Tensor);
protected:
  vector<Index> shape_;

  // So we don't have to put 'this->' everywhere
  using Buffer<Backend>::backend_;
  using Buffer<Backend>::type_;
  using Buffer<Backend>::owned_;
  using Buffer<Backend>::data_;
  using Buffer<Backend>::size_;
  using Buffer<Backend>::true_size_;
};
} // namespace ndll

#endif // NDLL_PIPELINE_DATA_TENSOR_H_
