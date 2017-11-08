#ifndef NDLL_PIPELINE_DATA_TENSOR_LIST_H_
#define NDLL_PIPELINE_DATA_TENSOR_LIST_H_

#include <cstring>

#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/buffer.h"

namespace ndll {

typedef vector<Index> Dims;

/**
 * @brief Stores a number of Tensors in a contiguous buffer. 
 * Functions similar to a jagged tensor, i.e. a tensor
 * where each element along the outer dimension can be of
 * different size.
 *
 * Provides helper functions for accessing individual Tensors
 * in the list.
 */
template <typename Backend>
class TensorList : public Buffer<Backend> {
public:
  TensorList() {}
  ~TensorList() = default;

  /**
   * @brief Resizes this TensorList to match the shape of the input.
   */
  template <typename InBackend>
  inline void ResizeLike(const TensorList<InBackend> &other) {
    Resize(other.shape_);
  }

  /**
   * @brief Copies the input TensorList, resizing this TensorList and 
   * changing the underlying data type if needed.
   */
  template <typename SrcBackend>
  inline void Copy(const TensorList<SrcBackend> &other, cudaStream_t stream) {
    this->set_type(other.type());
    ResizeLike(other);
    type_.Copy<Backend, SrcBackend>(this->raw_mutable_data(),
        other.raw_data(), this->size(), stream);
  }
  
  /**
   * @brief Resize function to allocate a list of tensors. The input vector
   * contains a set of dimensions for each tensor to be allocated in the
   * list.
   */
  inline void Resize(const vector<Dims> &new_shape) {
    if (new_shape == shape_) return;
    
    // Calculate the new size
    Index num_tensor = new_shape.size(), new_size = 0;
    offsets_.resize(num_tensor);
    for (Index i = 0; i < num_tensor; ++i) {
      auto tensor_size = Product(new_shape[i]);

      // Save the offset of the current sample & accumulate the size
      offsets_[i] = new_size;
      new_size += tensor_size;
    }
    NDLL_ENFORCE(new_size >= 0, "Invalid negative buffer size.");

    // Resize the underlying allocation and save the new shape
    ResizeHelper(new_size);
    shape_ = new_shape;
  }

  /**
   * @brief Wraps the data owned by the input TensorList. Both lists must
   * have valid types, and the input list must have enough storage to
   * store a single element of the calling objects type.
   *
   * When this function is called, the calling object shares the 
   * underlying allocation of the input TensorList. Its size is reset
   * to be the maximum number of elements of its type that can be
   * stored in the input lists allocation. While this list shares
   * data with another list, 'shares_data()' will return 'true'.
   */
  inline void ShareData(const TensorList<Backend> &other) {
    NDLL_ENFORCE(IsValidType(this->type_), "To share data another "
        "TensorLists data, a TensorList must have a valid data type.");
    NDLL_ENFORCE(IsValidType(other.type_), "To share data, "
        "the input TensorList must have a valid data type");

    // Find the maximum elements of our type we can store
    // in the shared allocated buffer
    Index possible_elements = other.num_bytes_ / this->type_.size();
    NDLL_ENFORCE(possible_elements > 0, "Shared data size smaller than "
        "a single element of other TensorLists type: " +
        std::to_string(other.num_bytes_) + " v. " +
        std::to_string(this->type_.size()));

    // Set our size to the maximum number of possible elements of our
    // type we can store in the shared buffer.
    size_ = possible_elements;
    shape_ = {{(Index)size_}}; // default size
    offsets_ = {(Index)0}; // offset for single tensor
    
    // Save the underlying allocation pointer and size
    data_ = other.data_;
    num_bytes_ = other.num_bytes_;
    shares_data_ = true;
  }
  
  /**
   * @brief Returns a typed pointer to the tensor with the given index.
   */
  template <typename T>
  inline T* mutable_tensor(int idx) {
    return this->template mutable_data<T>() + tensor_offset(idx);
  }

  /**
   * @brief Returns a const typed pointer to the tensor with the given index.
   */
  template <typename T>
  inline const T* tensor(int idx) const {
    return this->template data<T>() + tensor_offset(idx);
  }
  
  /**
   * @brief Returns a raw pointer to the tensor with the given index.
   */
  inline void* raw_mutable_tensor(int idx) {
    return static_cast<void*>(
        static_cast<uint8*>(this->raw_mutable_data()) +
        (tensor_offset(idx) * type_.size())
        );
  }

  /**
   * @brief Returns a const raw pointer to the tensor with the given index.
   */
  inline const void* raw_tensor(int idx) const {
    return static_cast<const void*>(
        static_cast<const uint8*>(this->raw_data()) +
        (tensor_offset(idx) * type_.size())
        );
  }
  
  /**
   * @brief Returns the number of tensors in the list.
   */
  inline int ntensor() const {
    return shape_.size();
  }

  /**
   * @brief Returns the offset of the tensor with the given index.
   */
  inline Index tensor_offset(int idx) const {
#ifndef NDEBUG
    NDLL_ENFORCE(idx >= 0, "Negative index not supported");
    NDLL_ENFORCE((size_t)idx < offsets_.size(), "Index out of offset range");
#endif
    return offsets_[idx];
  }

  /**
   * @brief Return the shape of the tensor with the given index.
   */
  inline vector<Index> tensor_shape(int idx) const {
#ifndef NDEBUG
    NDLL_ENFORCE(idx >= 0, "Negative index not supported");
    NDLL_ENFORCE((size_t)idx < shape_.size(), "Index out of offset range");
#endif
    return shape_[idx];
  }

  /**
   * @brief Returns a bool indicating if the list shares its underlying storage.
   */
  inline bool shares_data() const { return shares_data_; }
  
  // So we can access the members of other TensorListes
  // with different template types
  template <typename InBackend>
  friend class TensorList;
  
  DISABLE_COPY_MOVE_ASSIGN(TensorList);
protected:
  // We store a set of dimension for each tensor in the list.
  // We also pre-compute the offsets of each tensor in the
  // underlying allocation for random access
  vector<Dims> shape_;
  vector<Index> offsets_;

  USE_BUFFER_MEMBERS();
};

} // namespace ndll

#endif // NDLL_PIPELINE_DATA_TENSOR_LIST_H_
