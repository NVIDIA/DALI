#ifndef NDLL_PIPELINE_DATA_BATCH_H_
#define NDLL_PIPELINE_DATA_BATCH_H_

#include "ndll/pipeline/data/buffer.h"

namespace ndll {

typedef vector<Index> Dims;

/**
 * @brief Stores a batch of 'N' samples. Supports batches of 
 * jagged samples, i.e. batches where all samples are not
 * the sample dimensions
 */
template <typename Backend>
class Batch : public Buffer<Backend> {
public:
  Batch() {}
  ~Batch() = default;

  /**
   * @brief Resizes this batch to match the shape of the input batch
   */
  template <typename InBackend>
  inline void ResizeLike(const Batch<InBackend> &other) {
    Resize(other.batch_shape_);
  }
  
  /**
   * @brief Resize function to create jagged batches. The outer vector
   * size is taken to be the samples dimension, i.e. N = shape.size();
   *
   * Note: this method calculate some meta-data on the jagged tensor for
   * later use. Calling it repeatedly will be of non-negligible cost.
   */
  inline void Resize(const vector<Dims> &shape) {
    NDLL_ENFORCE(shape.size() > 0, "Batches must have at least a single datum");
    NDLL_ENFORCE(owned_, "Buffer does not own underlying "
        "storage, calling 'Resize()' not allowed");
    if (shape == batch_shape_) return;

    // Calculate the new size
    Index new_size = 0;
    for (auto &vec : shape) {
      Index tmp = 1;
      for (auto &val : vec) tmp *= val;
      new_size += tmp;
    }

    if (type_.id() == NO_TYPE) {
      // If the type has not been set yet, we just set the size
      // and shape of the buffer and do not allocate any memory.
      // Any previous resize dims are overwritten.
      size_ = new_size;
      true_size_ = new_size;
      batch_shape_ = shape;
      CalculateOffsets();
      return;
    }
    
    if (new_size > true_size_) {
      // Re-allocate the buffer to meet the new size requirements
      backend_.Delete(data_, true_size_*type_.size());
      data_ = backend_.New(new_size*type_.size());
      true_size_ = new_size;
    }

    // If we have enough storage already allocated, don't reallocate
    size_ = new_size;
    batch_shape_ = shape;
    CalculateOffsets();
  }

  /**
   * @brief returns a raw pointer to the sample with the given index
   */
  inline void* raw_datum(int idx) {
    return static_cast<void*>(
        static_cast<uint8*>(this->raw_data()) +
        (datum_offset(idx) * type_.size())
        );
  }
  
  /**
   * @brief Returns the number of samples in the batch
   */
  inline int ndatum() const {
    return batch_shape_.size();
  }

  /**
   * @brief returns the offset of the sample with the given index
   */
  inline Index datum_offset(int idx) const {
#ifndef NDEBUG
    NDLL_ENFORCE((size_t)idx >= 0, "Negative index not supported");
    NDLL_ENFORCE((size_t)idx < offsets_.size(), "Index out of offset range");
#endif
    return offsets_[idx];
  }

  /**
   * @brief return the shape of the sample with the given index
   */
  inline vector<Index> datum_shape(int idx) const {
#ifndef NDEBUG
    NDLL_ENFORCE((size_t)idx >= 0, "Negative index not supported");
    NDLL_ENFORCE((size_t)idx < batch_shape_.size(), "Index out of offset range");
#endif
    return batch_shape_[idx];
  }

  // So we can access the members of other Batches
  // w/ different template types
  template <typename InBackend>
  friend class Batch;
  
  DISABLE_COPY_MOVE_ASSIGN(Batch);
protected:
  // Helper to calculate datum offsets
  void CalculateOffsets() {
    int batch_size = batch_shape_.size();
    offsets_.resize(batch_size);
    
    Index offset = 0;
    for (int i = 0; i < batch_size; ++i) {
      offsets_[i] = offset;
      offset += Product(batch_shape_[i]);
    }
  }
  
  // We maintain a vector of 'Dims' to allow us to store
  // jagged tensors.  We also cache the offset of each sample
  vector<Dims> batch_shape_;
  vector<Index> offsets_;

  // So we don't have to put 'this->' everywhere
  using Buffer<Backend>::backend_;
  using Buffer<Backend>::type_;
  using Buffer<Backend>::owned_;
  using Buffer<Backend>::data_;
  using Buffer<Backend>::size_;
  using Buffer<Backend>::true_size_;
};

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
    Reset(batch, sample_idx);
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
    shape_ = shape;
  }
  
  inline void Reset(Batch<Backend> *batch, int sample_idx) {
#ifndef NDEBUG
    NDLL_ENFORCE(batch != nullptr, "Input batch is nullptr");
    NDLL_ENFORCE(sample_idx >= 0, "Negative index not supported");
    NDLL_ENFORCE(sample_idx < batch->ndatum(), "Sample index out of range");
#endif
    
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

#endif // NDLL_PIPELINE_DATA_BATCH_H_
