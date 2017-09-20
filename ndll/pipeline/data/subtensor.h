#ifndef NDLL_PIPELINE_DATA_SUBTENSOR_H_
#define NDLL_PIPELINE_DATA_SUBTENSOR_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/tensor.h"

namespace ndll {

// SubTensors wrap a portion of a Tensor and provide access to the underlying
// data without enforcing the original type of the Tensor data that was wrapped.
// This enables us to pack data of different types into large Tensors and perform
// single copies to the GPU.
//
// We define separate SubTensor classes for CPU & GPU data, so that functions can
// enforce the type of underlying data
//
// TODO(tgale): SubTensors are simple and are free to be copied. They
// do not own their underlying data and thus no cleanup or memory
// copying must be performed. However, the tensor a subtensor wraps
// could go out of scope, leaving the subtensor in a nasty state.
// The pointer is offset, so shared pointers are not an option.
// Consider fixing this issue.

/**
 * Wraps a portion of a Tensor that stores host-side data
 */
class CPUSubTensor {
public:
  /**
   * Wraps the portion of the input 1D tensor from [in.data()+start, in.data()+end)
   */
  template <typename Backend>
  CPUSubTensor(Tensor<Backend> *in, Index start, Index end) {
    Reset(in, start, end);
  }

  /**
   * Wraps the portion of the input 1D tensor from [in.data()+start, in.data()+end)
   */
  template<typename Backend>
  inline typename std::enable_if<std::is_base_of<CPUBackend, Backend>::value>::type
  Reset(Tensor<Backend> *in, Index start, Index end) {
#ifndef NDEBUG
    NDLL_ENFORCE(start >= 0);
    NDLL_ENFORCE(end > start);
    NDLL_ENFORCE(end <= in.size());
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

/**
 * Wraps a portion of a Tensor that stores device-side data
 */
class GPUSubTensor {
public:
  /**
   * Wraps the portion of the input 1D tensor from [in.data()+start, in.data()+end)
   */
  template <typename Backend>
  GPUSubTensor(Tensor<Backend> *in, Index start, Index end) {
    Reset(in, start, end);
  }

  /**
   * Wraps the portion of the input 1D tensor from [in.data()+start, in.data()+end)
   */
  template<typename Backend>
  inline typename std::enable_if<std::is_base_of<GPUBackend, Backend>::value>::type
  Reset(Tensor<Backend> *in, Index start, Index end) {
#ifndef NDEBUG
    NDLL_ENFORCE(start >= 0);
    NDLL_ENFORCE(end > start);
    NDLL_ENFORCE(end <= in.size());
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

#endif // NDLL_PIPELINE_DATA_SUBTENSOR_H_

