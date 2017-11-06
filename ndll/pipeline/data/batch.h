#ifndef NDLL_PIPELINE_DATA_BATCH_H_
#define NDLL_PIPELINE_DATA_BATCH_H_

#include <cstring>

#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/buffer.h"

namespace ndll {

typedef vector<Index> Dims;

/**
 * @brief Stores a batch of 'N' samples. Supports batches of 
 * jagged samples, i.e. batches where all samples are not
 * the sample dimensions
 *
 * Batch objects conform to the type management system defined
 * in @ref Buffer.
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
    Resize(other.shape_);
  }

  /**
   * @brief Copies the input batch, resizing this batch and changing 
   * the underlying data type if needed
   */
  template <typename SrcBackend>
  inline void Copy(const Batch<SrcBackend> &other, cudaStream_t stream) {
    this->set_type(other.type());
    ResizeLike(other);
    type_.Copy<Backend, SrcBackend>(this->raw_mutable_data(),
        other.raw_data(), this->size(), stream);
  }

  /**
   * @brief Resizes the tensor to match the input dense dimensions.
   */
  inline void Resize(const vector<Index> &new_shape) {
    // Create the equivalent jagged shape and resize. If the calculation
    // of the offsets & resizing becomes too expensive in the pipeline,
    // we can re-write this method to not use the jagged resize method
    // and just set the offset based on the dense shape.


    // Note: Scalar values are represented as 0 dimensional Tensors.
    Index new_size = Product(new_shape);
    ResizeHelper(new_size);

    // Save the new Tensor shape
    shape_ = new_shape;

    // Note: Scalar values are represented as 0 dimensional Tensors.
    // If the input has 0 dims, we set the number of samples to be
    // one and set its offset to be 0
    Index batch_size = new_shape.size() > 0 ? new_shape[0] : 1;
    offsets_.resize(batch_size);
    
    // Calculate the offset of each sample in the buffer
    Index sample_size = new_size / batch_size;
    Index offset = 0;
    for (int i = 0; i < batch_size; ++i) {
      offsets_[i] = offset;
      offset += sample_size;
    }
  }
  
  /**
   * @brief Resize function to create batches. The outer vector
   * size is taken to be the samples dimension, i.e. N = new_shape.size();
   */
  inline void Resize(const vector<Dims> &new_shape) {
    NDLL_ENFORCE(new_shape.size() > 0, "Batches must have at least a single sample.");
    if (new_shape == shape_) return;

    // Calculate the new size
    Index new_size = 0;
    for (auto &vec : new_shape) {
      Index tmp = 1;
      for (auto &val : vec) tmp *= val;
      new_size += tmp;
    }
    NDLL_ENFORCE(new_size >= 0, "Invalid negative buffer size.");
    
    // Resize the underlying allocation
    ResizeHelper(new_size);

    // Set the tensor size
    shape_ = new_shape;

    // Calculate the offset of each sample in the buffer
    int batch_size = shape_.size();
    offsets_.resize(batch_size);
    
    Index offset = 0;
    for (int i = 0; i < batch_size; ++i) {
      offsets_[i] = offset;
      offset += Product(shape_[i]);
    }    
  }
  
  /**
   * @brief Wraps the data owned by the input Batch. Both batches must
   * have valid types, and the input batch must have enough storage to
   * store a single element of the calling objects type.
   *
   * When this function is called, the calling object shares the 
   * underlying allocation of the input batch. Its size is reset
   * to be the maximum number of elements of its type that can be
   * stored in the input batches allocation. While this batch shares
   * data with another batch, 'shares_data()' will return 'true'.
   */
  inline void ShareData(const Batch<Backend> &other) {
    NDLL_ENFORCE(IsValidType(this->type_), "To share data another "
        "batches data, a batch must have a valid data type.");
    NDLL_ENFORCE(IsValidType(other.type_), "To share data, "
        "the input batch must have a valid data type");

    // Find the maximum elements of our type we can store
    // in the shared allocated buffer
    Index possible_elements = other.num_bytes_ / this->type_.size();
    NDLL_ENFORCE(possible_elements > 0, "Shared data size smaller than "
        "a single element of other batches type: " +
        std::to_string(other.num_bytes_) + " v. " +
        std::to_string(this->type_.size()));

    // Set our size to the maximum number of possible elements of our
    // type we can store in the shared buffer.
    size_ = possible_elements;
    shape_ = {{(Index)size_}}; // default size
    offsets_ = {(Index)0}; // offset for single sample
    
    // Save the underlying allocation pointer and size
    data_ = other.data_;
    num_bytes_ = other.num_bytes_;
    shares_data_ = true;
  }
  
  /**
   * @brief Returns a typed pointer to the sample with the given index.
   */
  template <typename T>
  inline T* mutable_sample(int idx) {
    return this->template mutable_data<T>() + sample_offset(idx);
  }

  /**
   * @brief Returns a const typed pointer to the sample with the given index.
   */
  template <typename T>
  inline const T* sample(int idx) const {
    return this->template data<T>() + sample_offset(idx);
  }
  
  /**
   * @brief Returns a raw pointer to the sample with the given index.
   */
  inline void* raw_mutable_sample(int idx) {
    return static_cast<void*>(
        static_cast<uint8*>(this->raw_mutable_data()) +
        (sample_offset(idx) * type_.size())
        );
  }

  /**
   * @brief Returns a const raw pointer to the sample with the given index.
   */
  inline const void* raw_sample(int idx) const {
    return static_cast<const void*>(
        static_cast<const uint8*>(this->raw_data()) +
        (sample_offset(idx) * type_.size())
        );
  }
  
  /**
   * @brief Returns the number of samples in the batch.
   */
  inline int nsample() const {
    return shape_.size();
  }

  /**
   * @brief Returns the offset of the sample with the given index.
   */
  inline Index sample_offset(int idx) const {
#ifndef NDEBUG
    NDLL_ENFORCE(idx >= 0, "Negative index not supported");
    NDLL_ENFORCE((size_t)idx < offsets_.size(), "Index out of offset range");
#endif
    return offsets_[idx];
  }

  /**
   * @brief Return the shape of the sample with the given index.
   */
  inline vector<Index> sample_shape(int idx) const {
#ifndef NDEBUG
    NDLL_ENFORCE(idx >= 0, "Negative index not supported");
    NDLL_ENFORCE((size_t)idx < shape_.size(), "Index out of offset range");
#endif
    return shape_[idx];
  }

  /**
   * @brief Returns a bool indicating if the batch shares its underlying storage.
   */
  inline bool shares_data() const { return shares_data_; }
  
  // So we can access the members of other Batches
  // with different template types
  template <typename InBackend>
  friend class Batch;
  
  DISABLE_COPY_MOVE_ASSIGN(Batch);
protected:

  // Helper to resize the underlying allocation. Does not touch
  // the Tensors shape or offsets.
  inline void ResizeHelper(Index new_size) {
    NDLL_ENFORCE(new_size >= 0, "Input size less than zero not supported.");
    
    if (!IsValidType(type_)) {
      // If the type has not been set yet, we just set the size of the
      // buffer and do not allocate any memory. Any previous size is
      // overwritten.
      NDLL_ENFORCE(data_ == nullptr, "Buffer has no type, data_ should be nullptr.");
      NDLL_ENFORCE(num_bytes_ == 0, "Buffer has no type, num_bytes_ should be 0.");
      
      size_ = new_size;
      return;
    }

    size_t new_num_bytes = new_size * type_.size();
    if (new_num_bytes > num_bytes_) {
      data_.reset(Backend::New(new_num_bytes), std::bind(
              &Buffer<Backend>::DeleterHelper,
              this, std::placeholders::_1,
              type_, new_size));
      num_bytes_ = new_num_bytes;
      
      // Call the constructor for the underlying datatype
      type_.template Construct<Backend>(data_.get(), new_size);
      
      // If we were sharing data, we aren't anymore
      shares_data_ = false;
    }

    size_ = new_size;
  }
  
  // We maintain a vector of 'Dims' to allow us to store
  // jagged tensors.  We also cache the offset of each sample
  vector<Dims> shape_;
  vector<Index> offsets_;
  
  // So we don't have to put 'this->' everywhere
  using Buffer<Backend>::backend_;
  using Buffer<Backend>::type_;
  using Buffer<Backend>::data_;
  using Buffer<Backend>::size_;
  using Buffer<Backend>::shares_data_;
  using Buffer<Backend>::num_bytes_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_DATA_BATCH_H_
