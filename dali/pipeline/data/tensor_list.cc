// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/pipeline/data/tensor_list.h"

#include <utility>

#include "dali/pipeline/data/global_workspace.h"

namespace dali {

// Acquire a buffer from the global workspace.
// Should only ever be called from mutable_data
template <typename Backend>
void TensorList<Backend>::acquire_buffer() {
  //std::cout << "acquire buffer TL" << std::endl;
  // Can't allocate an invalid type
  if (!IsValidType(type_)) return;
  //std::cout << "I have valid type" << std::endl;

  // already have a buffer, nothing else to do
  if (buffer_.get() != nullptr) {
    //std::cout << "I already have a buffer" << std::endl;
    return;
  }

  //std::cout << "I did not have a buffer, getting one" << std::endl;

  size_t num_elems = 0;
  for (size_t i = 0; i < shape_.size(); ++i) {
    num_elems += Product(shape_[i]);
  }
  //std::cout << "I need " << num_elems << " elements" << std::endl;
  size_t elem_size = type_.size();

  size_t buffer_size = num_elems * elem_size;
  std::cout << "I need " << buffer_size << " bytes" << std::endl;

  if (buffer_size > 0) {
    buffer_ = std::move(
        GlobalWorkspace::Get()->template AcquireBuffer<Backend>(buffer_size, pinned_));
    DALI_ENFORCE(buffer_.get() != nullptr);
    buffer_->ResizeAndSetType(num_elems, type_);
  }
}


template <typename Backend>
void TensorList<Backend>::set_type(TypeInfo type) {
  type_ = type;

  // if we have a buffer, set new type. Otherwise leave.
  if (buffer_.get()) {
    buffer_->set_type(type);
  }

  // check if metadata change allows acquire
  acquire_buffer();
}

template <typename Backend>
void TensorList<Backend>::Resize(const vector<Dims> &new_shape) {
  shape_ = new_shape;
  // Calculate the new size
  Index num_tensor = new_shape.size(), new_size = 0;
  offsets_.resize(num_tensor);
  for (Index i = 0; i < num_tensor; ++i) {
    auto tensor_size = Product(new_shape[i]);

    // Save the offset of the current sample & accumulate the size
    offsets_[i] = new_size;
    new_size += tensor_size;
  }
  DALI_ENFORCE(new_size >= 0, "Invalid negative buffer size.");

  std::cout << "Resizing TL " << this << " to " << new_size << " bytes." << std::endl;
  // if we have a buffer already, resize it, otherwise store metadata and leave
  if (buffer_.get()) {
    buffer_->Resize(new_size);
  }
  std::cout << "Done resizing TL " << this << " to " << new_size << " bytes." << std::endl;

  std::cout << "TL " << this << " acquires buffer." << std::endl;
  // check if metadata change allows acquire
  acquire_buffer();

  // Tensor view of this TensorList is no longer valid
  if (tensor_view_) {
    tensor_view_->ShareData(this);
  }
}

template <typename Backend>
void TensorList<Backend>::release(cudaStream_t s) const {
  std::cout << "Releasing TL " << this << std::endl;
  std::cout << "Current TL ref count: " << reference_count_ << std::endl;
  if (reference_count_ == 0) {
    std::cout << "This TL cannot be released, ignoring" << std::endl;
    return;
  }
  reference_count_--;
  if (reference_count_ == 0) {
    std::cout << "Ref count reached 0, releasing to global workspace" << std::endl;
    if (s != nullptr) CUDA_CALL(cudaStreamSynchronize(s));

    if (shares_data_) return;

    GlobalWorkspace::Get()->ReleaseBuffer<Backend>(&buffer_, pinned_);

    buffer_.reset();
  }
}

template <typename Backend>
void TensorList<Backend>::force_release() {
  if (shares_data_) return;

  GlobalWorkspace::Get()->ReleaseBuffer<Backend>(&buffer_, pinned_);

  buffer_.reset();
}

template class TensorList<CPUBackend>;
template class TensorList<GPUBackend>;

}  // namespace dali
