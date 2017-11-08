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
  void Copy(const vector<T> &data, cudaStream_t stream) {
    this->template mutable_data<T>();
    this->Resize({(Index)data.size()});
    type_.Copy<Backend, CPUBackend>(this->raw_mutable_data(),
        data.data(), this->size(), stream);
  }
  
  /**
   * @brief Resizes the buffer to fit `Product(shape)` elements.
   * The underlying storage is only reallocated in the case that
   * the current buffer is not large enough for the requested 
   * number of elements.
   */
  inline virtual void Resize(const vector<Index> &shape) {
    Index new_size = Product(shape);
    ResizeHelper(new_size);
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

  USE_BUFFER_MEMBERS();
};

} // namespace ndll

#endif // NDLL_PIPELINE_DATA_TENSOR_H_
