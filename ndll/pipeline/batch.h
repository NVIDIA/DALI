#ifndef NDLL_PIPELINE_BATCH_H_
#define NDLL_PIPELINE_BATCH_H_

#include "ndll/pipeline/buffer.h"

namespace ndll {

typedef vector<Dim> Shape;

/**
 * @brief Stores a batch of 'N' samples. Supports batches of 
 * jagged samples, i.e. batches where all samples are not
 * the sample dimensions
 */
template <typename Backend>
class Batch : public Buffer<Backend> {
public:
  Batch() : jagged_(false) {}
  ~Batch() = default;

  // For base class 'Resize()' method
  using Buffer<Backend>::Resize;
  
  /**
   * @brief Basic resize function to create dense batches.
   * The 0th value in the input shape is assumed to be
   * the samples dimension, i.e. N = shape[0]
   */
  inline void Resize(const vector<Dim> &shape) {
    NDLL_ENFORCE(owned_, "Buffer does not own underlying "
        "storage, calling 'Resize()' not allowed");
    // This tensor is not jagged
    jagged_ = false; 
    offsets_.clear();
    
    int new_size = Product(shape);
    if (type_.id() == NO_TYPE) {
      // If the type has not been set yet, we just set the size
      // and shape of the buffer and do not allocate any memory.
      // Any previous resize dims are overwritten.
      size_ = new_size;
      true_size_ = new_size;
      batch_shape_ = {shape};
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
    batch_shape_ = {shape};
  }

  /**
   * @brief Resize function to create jagged batches. The outer vector
   * size is taken to be the samples dimension, i.e. N = shape.size();
   *
   * Note: this method calculate some meta-data on the jagged tensor for
   * later use. Calling it repeatedly will be of non-negligible cost.
   */
  inline void Resize(const vector<Shape> &shape) {
    if (shape == batch_shape_) return;
    NDLL_ENFORCE(owned_, "Buffer does not own underlying "
        "storage, calling 'Resize()' not allowed");
    // User called the wrong method, forward the shape to
    // the more non-jagged version
    if (shape.size() == 0) {
      Resize({});
      return;
    } else if (shape.size() == 1) {
      Resize(shape[0]);
      return;
    }
    jagged_ = true;

    // Calculate the new size
    int new_size = 1;
    for (auto &vec : shape) {
      for (auto &val : vec) {
        new_size *= val;
      }
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
        static_cast<uint8*>(data_) +
        (datum_offset(idx) * type_.size())
        );
  }
  
  /**
   * @brief Returns the number of samples in the batch
   */
  inline int ndatum() const {
    if (jagged_) {
      return batch_shape_.size();
    } else {
      return batch_shape_[0][0];
    }
  }

  /**
   * @brief returns the offset of the sample with the given index
   */
  inline Dim datum_offset(int idx) const {
#ifdef DEBUG
    NDLL_ENFORCE(idx > 0, "Negative index not supported");
    NDLL_ENFORCE(idx < offsets_.size(), "Index out of offset range");
#endif
    if (jagged_) {
      return offsets_[idx];
    } else {
      return idx * (Product(batch_shape_[0]) / batch_shape_[0][0]);
    }
  }

  /**
   * @brief return the shape of the sample with the given index
   */
  inline vector<Dim> datum_shape(int idx) const {
#ifdef DEBUG
    NDLL_ENFORCE(idx > 0, "Negative index not supported");
    NDLL_ENFORCE(idx < batch_shape_.size(), "Index out of offset range");
#endif
    if (jagged_) {
      return batch_shape_[idx];
    } else {
      return batch_shape_[0];
    }
  }
  
  inline bool jagged() const {
    if (batch_shape_.size() < 2) {
      return false;
    }
    return true;
  }
  
  DISABLE_COPY_MOVE_ASSIGN(Batch);
protected:
  // Helper to calculate datum offsets
  void CalculateOffsets() {
    int batch_size = batch_shape_.size();
    offsets_.resize(batch_size);
    
    int offset = 0;
    for (int i = 0; i < batch_size; ++i) {
      offsets_[i] = offset;
      offset += Product(batch_shape_[i]);
    }
  }
  
  // We maintain a vector of 'Shape's to allow us to store
  // jagged tensors. If the outer vector is of length 1,
  // the tensor is assumed to be dense. We also cache the
  // offsets of each sample in the jagged tensor. If the
  // Tensor is dense, these offsets are not created
  vector<Shape> batch_shape_;
  vector<Dim> offsets_;
  bool jagged_;

  // So we don't have to put 'this->' everywhere
  using Buffer<Backend>::backend_;
  using Buffer<Backend>::type_;
  using Buffer<Backend>::owned_;
  using Buffer<Backend>::data_;
  using Buffer<Backend>::size_;
  using Buffer<Backend>::true_size_;
};

/**
 * @brief Wraps a single datum from a Batch. Datum does
 * not own the underlying storage.
 */
template <typename Backend>
class Datum : Buffer<Backend> {
public:
  /**
   * @brief Construct a sub-buffer that wraps a single datum from 
   * the input buffer. Outer dimension of the buffer is assumed 
   * to be the samples dimension i.e. 'N'
   */
  inline Datum(Batch<Backend> *batch, int sample_idx) {
    Reset(batch, sample_idx);
  }

  inline void Reset(Batch<Backend> *batch, int sample_idx) {
#ifdef DEBUG
    NDLL_ENFORCE(sample_idx >= 0, "Negative index not supported");
    NDLL_ENFORCE(sample_idx < batch->ndatum(), "Sample index out of range");
#endif
    
    // The sub-batch does not own its memory
    owned_ = false;

    // Get the shape of this sample
    shape_ = batch->datum_shape(sample_idx);
    true_size_ = Product(shape_);
    size_ = true_size_;

    type_ = batch->type();
    data_ = batch->raw_datum(sample_idx);
  }
protected:
  // Stores the shape of the sample
  vector<Dim> shape_;

  // So we don't have to put 'this->' everywhere
  using Buffer<Backend>::backend_;
  using Buffer<Backend>::type_;
  using Buffer<Backend>::owned_;
  using Buffer<Backend>::data_;
  using Buffer<Backend>::size_;
  using Buffer<Backend>::true_size_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_BATCH_H_
