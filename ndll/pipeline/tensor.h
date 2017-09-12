#ifndef NDLL_PIPELINE_TENSOR_H_
#define NDLL_PIPELINE_TENSOR_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/buffer.h"

namespace ndll {

/**
 * @brief Stores dense, multi-dimensional data. Provides utilities methods
 * for handling dimensions and shapes of the stored data.
 */
template <typename Backend>
class Tensor : Buffer<Backend> {
public:
  Tensor() {}
  ~Tensor() = default;

  // For base class 'Resize()' method
  using Buffer<Backend>::Resize;
  
  /**
   * @brief Resizes the buffer to fit `Product(shape)` elements.
   * The underlying storage is only reallocated in the case that
   * the current buffer is not large enough for the requested 
   * number of elements.
   *
   * Passing in a shape of '{}' indicates a scalar value
   */
  inline virtual void Resize(const vector<Dim> &shape) {
    if (shape == shape_) return;
    int new_size = Product(shape);
    Resize(new_size);

    // Save the new shape
    shape_ = shape;
  }

  inline vector<Dim> shape() const {
    return shape_;
  }

  inline virtual int ndim() const {
    return shape_.size();
  }

  inline virtual Dim dim(int idx) const {
#ifdef DEBUG
    NDLL_ENFORCE(i < shape_.size(), "index exceeds ndim");
    NDLL_ENFORCE(i > 0, "negative index not supported");
#endif
    return shape_[idx];
  }
  
  DISABLE_COPY_MOVE_ASSIGN(Tensor);
protected:
  vector<Dim> shape_;
};
} // namespace ndll

#endif // NDLL_PIPELINE_TENSOR_H_
