// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_DATA_TENSOR_VECTOR_H_
#define DALI_PIPELINE_DATA_TENSOR_VECTOR_H_

#include <memory>
#include <vector>

#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"

#include "dali/kernels/tensor_shape.h"

namespace dali {

/**
 * @brief Merges TensorList<Backend> and std::vector<std::shared_ptr<Tensor<Backend>>> APIs
 * providing an uniform way of handling a collection/batch of tensors_.
 *
 * Propagates Buffer calls to every tensor uniformly
 *
 * @tparam Backend
 */
template <typename Backend>
class TensorVector {
 public:
  TensorVector() {}
  explicit TensorVector(int batch_size) : tensors_(batch_size, nullptr) {
    for (auto &t : tensors_) {
      t = std::make_shared<Tensor<Backend>>();
    }
  }

  TensorVector(const TensorVector &) = delete;
  TensorVector &operator=(const TensorVector &) = delete;

  auto operator[](size_t pos) {
    return tensors_[pos];
  }

  auto operator[](size_t pos) const {
    return tensors_[pos];
  }

  auto begin() noexcept {
    return tensors_.begin();
  }

  auto begin() const noexcept {
    return tensors_.begin();
  }

  auto cbegin() const noexcept {
    return tensors_.cbegin();
  }

  auto end() noexcept {
    return tensors_.end();
  }

  auto end() const noexcept {
    return tensors_.end();
  }

  auto cend() const noexcept {
    return tensors_.cend();
  }

  auto size() const noexcept {
    return tensors_.size();
  }

  kernels::TensorListShape<> shape() const {
    if (tensors_.empty()) {
      return {};
    }
    kernels::TensorListShape<> result(tensors_.size(), tensors_[0]->ndim());
    for (size_t i = 0; i < tensors_.size(); i++) {
      result.set_tensor_shape(i, tensors_[i]->shape());
    }
    return result;
  }

  DLL_PUBLIC inline void Resize(const kernels::TensorListShape<> &new_shape) {
    // N.B. There probably is nothing wrong with adjusting batchsize for give TensorVector
    // (sparse tensor list), but the semantics of what type and pinned status should
    // the new elements have or having some of them allocated and the new ones not
    // complicates the logic even further so we disallow it
    DALI_ENFORCE(tensors_.empty() || static_cast<int>(tensors_.size()) == new_shape.size(),
                 "Changing the batch size is prohibited. It should be set once.");
    if (tensors_.empty()) {
      allocate_tensors(new_shape.size());
    }
    for (int i = 0; i < new_shape.size(); i++) {
      tensors_[i]->Resize(new_shape[i]);
    }
  }

  inline void set_type(const TypeInfo &new_type) {
    type_ = new_type;
    for (auto t : tensors_) {
      t->set_type(new_type);
    }
  }

  inline TypeInfo type() const {
    // TODO(klecki): do we enforce the same type between elements, or prohibit mixing APIs?
    return type_;
  }

  inline void set_pinned(const bool pinned) {
    // Store the value, in case we pin empty vector and later call Resize
    pinned_ = pinned;
    for (auto &t : tensors_) {
      t->set_pinned(pinned);
    }
  }

  inline bool is_pinned() const {
    // TODO(klecki): do we enforce the same pin status between elements, or prohibit mixing APIs?
    bool pinned = true;
    for (auto &t : tensors_) {
      pinned = pinned && t->is_pinned();
    }
    return pinned_ && pinned;
  }

  /// @brief Reserved memory size is divided between all elements
  inline void reserve(size_t new_num_bytes) {
    reserved_batch_memory = new_num_bytes;
    // Won't do the division by 0 if there are no elements
    for (auto t : tensors_) {
      t->reserve(new_num_bytes / tensors_.size());
    }
  }

  inline void reserve(size_t new_num_bytes, int batch_size) {
    DALI_ENFORCE(tensors_.empty() || static_cast<int>(tensors_.size()) == batch_size,
                 "Changing the batch size is prohibited. It should be set once.");
    if (tensors_.empty()) {
      allocate_tensors(batch_size);
    }
    for (auto t : tensors_) {
      t->reserve(new_num_bytes);
    }
  }

 private:
  void allocate_tensors(int batch_size) {
    DALI_ENFORCE(tensors_.empty(), "Changing the batch size is prohibited. It should be set once.");
    tensors_.resize(batch_size, nullptr);
    for (auto &t : tensors_) {
      t = std::make_shared<Tensor<Backend>>();
      t->set_pinned(pinned_);
      t->reserve(reserved_batch_memory / batch_size);
      if (IsValidType(type_)) {
        t->set_type(type_);
      }
    }
  }
  std::vector<std::shared_ptr<Tensor<Backend>>> tensors_;
  // pinned status and type info should be uniform
  bool pinned_ = true;
  size_t reserved_batch_memory = 0;
  TypeInfo type_ = TypeInfo();
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_VECTOR_H_
