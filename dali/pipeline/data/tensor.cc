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

#include "dali/pipeline/data/tensor.h"

#include "dali/pipeline/data/global_workspace.h"

namespace dali {

template <typename Backend>
void Tensor<Backend>::acquire_buffer() {
  if (shares_data()) {
    buffer_.reset();
  }

  DALI_ENFORCE(IsValidType(type_),
      "Tensor needs to have a valid type before acquiring buffer.");

  size_t buffer_size = size() * type_.size();
  std::cout << "I need " << buffer_size << " bytes" << std::endl;

  // If we do not already have a buffer
  // we need to get one from the GlobalWorkspace
  if (buffer_.get() == nullptr && buffer_size > 0) {
    buffer_ = std::move(
        GlobalWorkspace::Get()->template AcquireBuffer<Backend>(buffer_size, pinned_));
    DALI_ENFORCE(buffer_.get() != nullptr);
  }

  if (buffer_.get() != nullptr) {
    buffer_->Resize(buffer_size);
  }
}

template <typename Backend>
void Tensor<Backend>::set_type_and_size(TypeInfo new_type, const vector<Index> &new_shape) {
  DALI_ENFORCE(IsValidType(new_type), "new_type must be valid type.");
  set_shape(new_shape);
  type_ = new_type;
  acquire_buffer();
}

template <typename Backend>
void Tensor<Backend>::set_shape(const vector<Index> &new_shape) {
  if (shape_ == new_shape) return;

  shape_ = new_shape;

  DALI_ENFORCE(size() >= 0, "Invalid negative buffer size.");
}

template <typename Backend>
inline void Tensor<Backend>::Resize(const vector<Index> &shape) {
  set_shape(shape);

  if (IsValidType(type_)) {
    acquire_buffer();
  }
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

    if (!shares_data()) {
      GlobalWorkspace::Get()->ReleaseBuffer<Backend>(&buffer_, pinned_);
    }
    buffer_.reset();
  }
}

template <typename Backend>
void Tensor<Backend>::force_release() {
  if (!shares_data()) {
    GlobalWorkspace::Get()->ReleaseBuffer<Backend>(&buffer_, pinned_);
  }
  buffer_.reset();
}

template class Tensor<GPUBackend>;
template class Tensor<CPUBackend>;
}  // namespace dali
