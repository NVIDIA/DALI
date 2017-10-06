#ifndef NDLL_PIPELINE_DATA_SUB_TENSOR_H_
#define NDLL_PIPELINE_DATA_SUB_TENSOR_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/tensor.h"

namespace ndll {

/**
 * @brief Wraps a portion of a Tensor. Does not own its underlying storage.
 *
 * SubTensors wrap a portion of a Tensor and provide access to the underlying
 * data without enforcing the original type of the Tensor data that was wrapped.
 * This enables us to pack data of different types into large Tensors and perform
 * single copies to the GPU.
 */
template <typename Backend>
class SubTensor {
public:
  SubTensor() : data_(nullptr), size_(0) {}
  
  /**
   * Wraps the portion of the input 1D tensor from [in.data()+start, in.data()+end)
   */
  SubTensor(Tensor<Backend> *in, Index start, Index end) {
    Reset(in, start, end);
  }

  /**
   * Wraps the portion of the input 1D tensor from [in.data()+start, in.data()+end)
   */
  void Reset(Tensor<Backend> *in, Index start, Index end) {
#ifndef NDEBUG
    NDLL_ENFORCE(start >= 0);
    NDLL_ENFORCE(end > start);
    NDLL_ENFORCE(end <= in->size());
#endif
    NDLL_ENFORCE(in->ndim() == 1, "This method only supports wrapping in 1-dimension");
    NDLL_ENFORCE(in != nullptr);

    // The sub-tensor does not own its memory
    shape_ = {end-start};
    size_ = end-start;

    // Calling raw_data here will enforce that the Tensor
    // has valid type and is thus allocated
    data_ = static_cast<void*>(
        static_cast<uint8*>(in->raw_data())
        + start*in->type().size()
        );
  }

  template <typename T>
  inline T* data() {
    return static_cast<T*>(data_);
  }

  template <typename T>
  inline const T* data() const {
    return static_cast<T*>(data_);
  }

  Index size() const {
    return size_;
  }

  vector<Index> shape() const {
    return shape_;
  }

protected:
  void *data_;
  vector<Index> shape_;
  Index size_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_DATA_SUB_TENSOR_H_

