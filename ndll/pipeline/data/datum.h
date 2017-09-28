#ifndef NDLL_PIPELINE_DATA_DATUM_H_
#define NDLL_PIPELINE_DATA_DATUM_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/batch.h"

namespace ndll {

/**
 * @brief Datum can either allocate its own storage or 
 * wrap a single datum from a batch.
 */
template <typename Backend>
class Datum : public Buffer<Backend> {
public:
  /**
   * @brief Creates a default Datum that holds no data
   */
  inline Datum() {}

  /**
   * @brief Creates a Datum object with the input shape
   */
  inline Datum(const vector<Index> &shape) {
    Resize(shape);
  }
  
  /**
   * @brief Construct a sub-buffer that wraps a single datum from 
   * the input buffer. Outer dimension of the buffer is assumed 
   * to be the samples dimension i.e. 'N'
   */
  inline Datum(Batch<Backend> *batch, int sample_idx) {
    WrapSample(batch, sample_idx);
  }

  /**
   * @brief Resizes the Datum. If the Datum is currently wrapping data
   * that it does not own, it will detach from the wrapped data and
   * allocate its own data.
   */
  inline void Resize(const vector<Index> &shape) {
    if (!owned_) {
      // Reset to a default state
      data_ = nullptr;
      shape_.clear();
      true_size_ = 0;
      size_ = 0;

      TypeMeta new_type;
      type_ = new_type;
      owned_ = true;
    }

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
    size_ = new_size;
    shape_ = shape;
  }

  /**
   * @brief convenience method that calls 'Resize(const vector<Index> &shape)'
   */
  inline void ResizeLike(const Datum<Backend> &other) {
    Resize(other.shape());
  }

  /**
   * @brief Copies the data from the input Datum
   */
  inline void Copy(const Datum<Backend> &other) {
    this->set_type(other.type());
    this->ResizeLike(other);
    MemCopy(this->raw_data(), other.raw_data(), other.nbytes());
  }
  
  inline void WrapSample(Batch<Backend> *batch, int sample_idx) {
#ifndef NDEBUG
    NDLL_ENFORCE(sample_idx >= 0, "Negative index not supported");
    NDLL_ENFORCE(sample_idx < batch->ndatum(), "Sample index out of range");
#endif
    NDLL_ENFORCE(batch != nullptr, "Input batch is nullptr");

    if (owned_ && true_size_ > 0) {
      // Clean up the current underlying storage
      backend_.Delete(data_, true_size_*type_.size());

      // Set back to default state
      data_ = nullptr;
      size_ = 0;
      true_size_ = 0;
      shape_.clear();
      TypeMeta new_type;
      type_ = new_type;
    }
    
    // The sub-batch does not own its memory
    owned_ = false;

    // Get the shape of this sample
    shape_ = batch->datum_shape(sample_idx);
    true_size_ = Product(shape_);
    size_ = true_size_;

    // Calling raw_datum here will enforce that the type is valid
    type_ = batch->type();
    data_ = batch->raw_datum(sample_idx);
  }

  /**
   * @brief get the shape of the datum
   */
  inline vector<Index> shape() const {
    return shape_;
  }
  
  DISABLE_COPY_MOVE_ASSIGN(Datum);
protected:
  // Stores the shape of the sample
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

#endif // NDLL_PIPELINE_DATA_DATUM_H_
