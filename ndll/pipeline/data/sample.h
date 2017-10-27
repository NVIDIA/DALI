#ifndef NDLL_PIPELINE_DATA_SAMPLE_H_
#define NDLL_PIPELINE_DATA_SAMPLE_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/batch.h"

namespace ndll {

/**
 * @brief Sample can either allocate its own storage or  wrap 
 * a single sample from a batch.
 *
 * In the case that a Sample object is wrapping memory that it
 * does not own, methods that can trigger memory allocation
 * will cause the Sample to detach from the wrapped memory and
 * allocate its own underlying storage. These methods are @n
 * 'set_type()' - Will allocate memory if the calling type does 
 * not match the underlying type of the buffer @n
 * 'mutable_data<T>()' - Calls 'set_type' internally @n
 * 'Resize()' - Detaches from the wrapped memory and allocates
 * memory for the input number of elements.
 */
template <typename Backend>
class Sample : public Buffer<Backend> {
public:
  /**
   * @brief Creates a default Sample that holds no data
   */
  inline Sample() : owned_(true) {}

  ~Sample() = default;
  
  /**
   * @brief Creates a Sample object with the input shape
   */
  inline Sample(const vector<Index> &shape) {
    Resize(shape);
  }
  
  /**
   * @brief Construct a sub-buffer that wraps a single sample from 
   * the input buffer. Outer dimension of the buffer is assumed 
   * to be the samples dimension i.e. 'N'
   */
  inline Sample(Batch<Backend> *batch, int sample_idx) {
    WrapSample(batch, sample_idx);
  }

  /**
   * @brief Resizes the Sample. If the Sample is currently wrapping data
   * that it does not own, it will detach from the wrapped data and
   * allocate its own data.
   */
  inline void Resize(const vector<Index> &shape) {
    if (!owned_) {
      // Reset to a default state
      data_.reset();
      shape_.clear();
      num_bytes_ = 0;
      size_ = 0;

      TypeInfo new_type;
      type_ = new_type;
      owned_ = true;
    }

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
   * @brief Convenience method sizes the Sample to match the input Sample
   */
  inline void ResizeLike(const Sample<Backend> &other) {
    Resize(other.shape());
  }

  /**
   * @brief Copies the data from the input Sample
   */
  void Copy(const Sample<Backend> &other, cudaStream_t stream = 0);

  /**
   * @brief Wraps the sample with the given index in the batch.
   * If the Sample owns its own underlying storage, the memory
   * is deallocated.
   */
  inline void WrapSample(Batch<Backend> *batch, int sample_idx) {
#ifndef NDEBUG
    NDLL_ENFORCE(sample_idx >= 0, "Negative index not supported");
    NDLL_ENFORCE(sample_idx < batch->nsample(), "Sample index out of range");
#endif
    NDLL_ENFORCE(batch != nullptr, "Input batch is nullptr");

    if (owned_ && num_bytes_ > 0) {
      // Set back to default state
      data_.reset();
      size_ = 0;
      num_bytes_ = 0;
      shape_.clear();
      TypeInfo new_type;
      type_ = new_type;
    }
    
    // The sample does not own its memory
    owned_ = false;

    // Get the shape of this sample
    shape_ = batch->sample_shape(sample_idx);
    size_ = Product(shape_);

    // Calling raw_mutable_sample here will enforce that the type is valid
    type_ = batch->type();
    data_.reset(batch->raw_mutable_sample(sample_idx),
        [](void *p) { /* noop: do not delete ptr in the middle of an allocation */ });

    // Note: In the case that the Sample does not own its underlying storage,
    // we keep the value of num_bytes_ equal to zero, indicating that the
    // sample has allocated no memory of its own.
    num_bytes_ = 0;
  }

  /**
   * @brief get the shape of the sample
   */
  inline vector<Index> shape() const {
    return shape_;
  }

  /**
   * @brief Returns a bool indicating if the Sample object owns its 
   * underlying storage.
   */
  inline bool owned() const {
    return owned_;
  }
  
  DISABLE_COPY_MOVE_ASSIGN(Sample);
protected:
  // Stores the shape of the sample
  vector<Index> shape_;

  // Indicates whether the Sample
  // owns it underlying storage
  bool owned_;
  
  using Buffer<Backend>::backend_;
  using Buffer<Backend>::type_;
  using Buffer<Backend>::data_;
  using Buffer<Backend>::size_;
  using Buffer<Backend>::num_bytes_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_DATA_SAMPLE_H_
