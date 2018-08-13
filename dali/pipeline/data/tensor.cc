// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/pipeline/data/tensor.h"

#include "dali/pipeline/data/global_workspace.h"

namespace dali {

template <typename Backend>
void Tensor<Backend>::acquire_buffer() {
  // not a valid type, don't try to acquire
  if (!IsValidType(type_)) return;

  // already have a buffer, nothing else to do
  if (buffer_.get() != nullptr) {
    buffer_->ResizeAndSetType(Product(shape_), type_);
    return;
  }

  auto num_elems = Product(shape_);
  auto elem_size = type_.size();

  auto buffer_size = num_elems * elem_size;

  if (buffer_size > 0) {
    buffer_ = std::move(
        GlobalWorkspace::Get()->template AcquireBuffer<Backend>(buffer_size, pinned_));
    DALI_ENFORCE(buffer_.get() != nullptr);
    buffer_->ResizeAndSetType(num_elems, type_);
  }
}

template <typename Backend>
void Tensor<Backend>::set_type(TypeInfo type) {
  type_ = type;

  // If we don't have a buffer already, get
  if (buffer_.get()) {
    buffer_->set_type(type);
  }

  // If we didn't already have a buffer, acquire one now
  acquire_buffer();
}

template <typename Backend>
inline void Tensor<Backend>::Resize(const vector<Index> &shape) {
  shape_ = shape;

  if (buffer_.get()) {
    buffer_->Resize(Product(shape_));
  }

  // If we didn't already have a buffer, acquire one now
  acquire_buffer();
}

template <typename Backend>
void Tensor<Backend>::release(cudaStream_t s) const {
  std::cout << "Releasing T " << this << std::endl;
  std::cout << "Current T ref count: " << reference_count_ << std::endl;
  if (reference_count_ == 0) {
    std::cout << "This T cannot be released, ignoring" << std::endl;
    return;
  }
  reference_count_--;
  if (reference_count_ == 0) {
    if (s != nullptr) CUDA_CALL(cudaStreamSynchronize(s));

    // If we're sharing data from somewhere else, no need to release
    // the buffer, just delete it - actual allcoations are handled
    // elsewhere
    if (shares_data_) {
      return;
    }
    // Release the buffer back into the global pool for re-use
    GlobalWorkspace::Get()->ReleaseBuffer<Backend>(&buffer_, pinned_);

    buffer_.reset();
  }
}

template <typename Backend>
void Tensor<Backend>::force_release() {
  if (shares_data_) {
    return;
  }
  // Release the buffer back into the global pool for re-use
  GlobalWorkspace::Get()->ReleaseBuffer<Backend>(&buffer_, pinned_);

  buffer_.reset();
}

template class Tensor<GPUBackend>;
template class Tensor<CPUBackend>;
}  // namespace dali
