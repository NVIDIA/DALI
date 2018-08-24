// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dali/pipeline/data/tensor_list.h"

#include <utility>

#include "dali/pipeline/data/global_workspace.h"

namespace dali {

// Acquire a buffer from the global workspace.
// Should only ever be called from mutable_data
template <typename Backend>
void TensorList<Backend>::acquire_buffer() {
  if (shares_data()) {
    buffer_.reset();
  }

  DALI_ENFORCE(IsValidType(type_),
      "TensorList needs to have a valid type before acquiring buffer.");

  size_t buffer_size = size() * type_.size();

  // If we do not already have a buffer
  // we need to get one from the GlobalWorkspace
  if (buffer_.get() == nullptr && buffer_size > 0) {
  std::cout << "[" << std::this_thread::get_id() << "] " << "I need " << buffer_size << " bytes and don't have a buffer yet" << std::endl;
    buffer_ = std::move(
        GlobalWorkspace::Get()->template AcquireBuffer<Backend>(buffer_size, pinned_));
    DALI_ENFORCE(buffer_.get() != nullptr);
  }

  if (buffer_.get() != nullptr) {
    bool changed = buffer_->Resize(buffer_size);

    // Tensor view of this TensorList is no longer valid
    if (tensor_view_ && changed) {
      tensor_view_->ShareData(this);
    }
  }
}


template <typename Backend>
void TensorList<Backend>::set_type_and_size(TypeInfo new_type, const vector<Dims> &new_shape) {
  DALI_ENFORCE(IsValidType(new_type), "new_type must be valid type.");
  set_shape(new_shape);
  type_ = new_type;
  acquire_buffer();
}

template <typename Backend>
void TensorList<Backend>::set_shape(const vector<Dims> &new_shape) {
  if (shape_ == new_shape) return;

  shape_ = new_shape;

  Index num_tensor = new_shape.size(), new_size = 0;
  offsets_.resize(num_tensor);
  for (Index i = 0; i < num_tensor; ++i) {
    auto tensor_size = Product(new_shape[i]);

    // Save the offset of the current sample & accumulate the size
    offsets_[i] = new_size;
    new_size += tensor_size;
  }
  size_ = new_size;
  DALI_ENFORCE(size_ >= 0, "Invalid negative buffer size.");
}

template <typename Backend>
void TensorList<Backend>::Resize(const vector<Dims> &new_shape) {
  set_shape(new_shape);

  std::cout << "Resizing TL " << this << " to " << size_ << " elements." << std::endl;

  if (IsValidType(type_)) {
    std::cout << "TL " << this << " acquires buffer." << std::endl;
    acquire_buffer();
  }
  std::cout << "Done resizing TL " << this << " to " << size_ << " elements." << std::endl;
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

    if (!shares_data()) {
      GlobalWorkspace::Get()->ReleaseBuffer<Backend>(&buffer_, pinned_);
    }
    buffer_.reset();
  }
}

template <typename Backend>
void TensorList<Backend>::force_release() {
  if (!shares_data()) {
    GlobalWorkspace::Get()->ReleaseBuffer<Backend>(&buffer_, pinned_);
  }
  buffer_.reset();
}

template class TensorList<CPUBackend>;
template class TensorList<GPUBackend>;

}  // namespace dali
