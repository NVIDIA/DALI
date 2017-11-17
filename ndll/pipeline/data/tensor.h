#ifndef NDLL_PIPELINE_DATA_TENSOR_H_
#define NDLL_PIPELINE_DATA_TENSOR_H_

#include <cstring>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/buffer.h"
#include "ndll/pipeline/data/tensor_list.h"

namespace ndll {

/**
 * @brief Stores dense, multi-dimensional data. Provides utilities 
 * methods for handling dimensions and shapes of the stored data.
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

  template <typename InBackend>
  void ResizeLike(const Tensor<InBackend> &other) {
    Resize(other.shape());
  }
  
  /**
   * @brief Resizes the buffer to fit `Product(shape)` elements.
   * The underlying storage is only reallocated in the case that
   * the current buffer is not large enough for the requested 
   * number of elements.
   */
  inline void Resize(const vector<Index> &shape) {
    Index new_size = Product(shape);
    ResizeHelper(new_size);
    shape_ = shape;
  }
  
  /**
   * @brief Wraps the data owned by the tensor at the given index
   * in the input tensor list. The input tensor list must have
   * a valid type, and the given index must be in the valid range
   * [0, tl.ntensor()).
   *
   * If sucessful, the tensor object will wrap the target data and
   * assume the datatype of the data stored in the TensorList.
   *
   * Because we are storing the pointer of the TensorList at an
   * offset, we do not guarantee that this allocation will persist
   * until both the owner and the sharer are finished with it. Thus,
   * it is up to the user to manage the scope of the sharing objects
   * to ensure correctness.
   */
  inline void ShareData(TensorList<Backend> *tl, int idx) {
    NDLL_ENFORCE(tl != nullptr, "Input TensorList is nullptr");
    NDLL_ENFORCE(IsValidType(tl->type()), "To share data, "
        "the input TensorList must have a valid data type.");
    NDLL_ENFORCE(idx >= 0, "Negative tensor index not supported.");
    NDLL_ENFORCE(idx < tl->ntensor(), "Index of " + std::to_string(idx) +
        " out of range for TensorList of size " + std::to_string(tl->ntensor()));
    // TODO(tgale): If we wanted to ensure the allocation is not cleaned up
    // while this object still uses it, we could just keep a copy of
    // the actual shared_ptr of the TensorList. Is this behavior something
    // that we are interested in supporting?
    
    // Reset our pointer to the correct offset inside the tensor list.
    // This is not the beginning of the allocation, so we pass a noop
    // deleter to the shared_ptr
    data_.reset(tl->raw_mutable_tensor(idx), [](void *) {});

    // Get the meta-data for the target tensor
    shape_ = tl->tensor_shape(idx);
    size_ = Product(shape_);
    type_ = tl->type();
    num_bytes_ = type_.size() * size_;
    shares_data_ = true;
  }

  /**
   * @brief Wraps the raw allocation. The input pointer must not be nullptr.
   * if the size of the allocation is zero, the Tensor is reset to a default
   * state and is NOT marked as sharing data.
   *
   * After wrapping the allocation, the Tensors size is set to 0, and its
   * type is reset to NoType. Future calls to Resize or setting of the 
   * Tensor type will evaluate whether or not the current allocation is
   * large enough to be used and proceed appropriately.
   *
   * The Tensor object assumes no ownership of the input allocation, and will
   * not de-allocate it when it is done using it. It is up to the user to
   * manage the lifetime of the allocation such that it persist while it is
   * in use by the Tensor.
   */
  inline void ShareData(void *ptr, size_t bytes) {
    NDLL_ENFORCE(ptr != nullptr, "Input pointer must not be nullptr.");

    // Save our new pointer and bytes. Reset our type, shape, and size
    data_.reset(ptr, [](void *) {});
    num_bytes_ = bytes;
    type_ = TypeInfo::Create<NoType>();
    shape_.clear();
    size_ = 0;

    // If the input pointer stores a non-zero size allocation, mark
    // that we are sharing our underlying data
    if (num_bytes_ > 0) shares_data_ = true;
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
