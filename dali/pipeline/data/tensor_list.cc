// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/pipeline/data/tensor_list.h"

#include <utility>

#include "dali/pipeline/data/global_workspace.h"

namespace dali {

// Acquire a buffer from the global workspace.
// Should only ever be called from mutable_data
template <typename Backend>
void TensorList<Backend>::acquire_buffer() {
  // Can't allocate an invalid type
  if (!IsValidType(type_)) return;

  // already have a buffer, nothing else to do
  if (buffer_.get() != nullptr) {
    return;
  }

  auto num_elems = 0;
  for (size_t i = 0; i < shape_.size(); ++i) {
    num_elems += Product(shape_[i]);
  }
  auto elem_size = type_.size();

  auto buffer_size = num_elems * elem_size;

  if (buffer_size > 0) {
    buffer_ = std::move(
        GlobalWorkspace::Get()->template AcquireBuffer<Backend>(buffer_size, pinned_));
    DALI_ENFORCE(buffer_.get() != nullptr);
    buffer_->set_type(type_);
    buffer_->Resize(num_elems);
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

  // if we have a buffer already, resize it, otherwise store metadata and leave
  if (buffer_.get()) {
    buffer_->Resize(new_size);
  }

  // check if metadata change allows acquire
  acquire_buffer();

  // Tensor view of this TensorList is no longer valid
  if (tensor_view_) {
    tensor_view_->ShareData(this);
  }
}

template <typename Backend>
void TensorList<Backend>::release(cudaStream_t s) const {
  reference_count_--;
  if (reference_count_ == 0) {
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
