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

#include <atomic>
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
  TensorVector() : views_count_(0), tl_(std::make_shared<TensorList<Backend>>()) {}

  explicit TensorVector(int batch_size)
      : views_count_(0), tensors_(batch_size, nullptr), tl_(std::make_shared<TensorList<Backend>>(batch_size)) {
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
    if (state_ == State::contiguous) {
      return tl_->shape();
    }
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
    if (state_ == State::contiguous) {
      tl_->Resize(new_shape);
      validate_views();
      return;
    }
    for (int i = 0; i < new_shape.size(); i++) {
      tensors_[i]->Resize(new_shape[i]);
    }
  }

  inline void set_type(const TypeInfo &new_type) {
    type_ = new_type;
    tl_->set_type(new_type);
    for (auto t : tensors_) {
      t->set_type(new_type);
    }
    validate_views();
  }

  inline TypeInfo type() const {
    if (state_ == State::contiguous) {
      return tl_->type();
    }
    if (tensors_.size() > 0) {
      return tensors_[0]->type();
    }
    return type_;
  }

  inline void set_pinned(const bool pinned) {
    // Store the value, in case we pin empty vector and later call Resize
    pinned_ = pinned;
    tl_->set_pinned(pinned);
    for (auto &t : tensors_) {
      t->set_pinned(pinned);
    }
  }

  inline bool is_pinned() const {
    // TODO(klecki): do we enforce the same pin status between elements, or prohibit mixing APIs?
    if (state_ == State::contiguous) {
      return tl_->is_pinned();
    }
    if (tensors_.size() > 0) {
      return tensors_[0]->is_pinned();
    }
    return pinned_;
  }

  /// @brief Reserve as contiguous tensor list internally
  inline void reserve(size_t total_bytes) {
    state_ = State::contiguous;
    tl_->reserve(total_bytes);
  }

  /// @brief Reserve as vector of `batch_size` tensors internally
  inline void reserve(size_t bytes_per_sample, int batch_size) {
    DALI_ENFORCE(tensors_.empty() || static_cast<int>(tensors_.size()) == batch_size,
                 "Changing the batch size is prohibited. It should be set once.");
    state_ = State::noncontiguous;
    // If we didn't declare the batch size but tried to pin memory or set type
    // we need to apply it to tensors
    if (tensors_.empty()) {
      allocate_tensors(batch_size);
    }
    for (auto t : tensors_) {
      t->reserve(bytes_per_sample);
    }
  }

  bool is_contiguous() {
    // TODO(klecki): check the views_count as well?
    return state_ == State::contiguous && views_count_ == size();
  }

 private:
  enum class State {
    contiguous,
    noncontiguous
  };
  void allocate_tensors(int batch_size) {
    DALI_ENFORCE(tensors_.empty(), "Changing the batch size is prohibited. It should be set once.");
    // If we didn't declare the batch size but tried to pin memory or set type
    // we need to apply it to tensors
    tensors_.resize(batch_size, nullptr);
    for (auto &t : tensors_) {
      t = std::make_shared<Tensor<Backend>>();
      t->set_pinned(pinned_);
      if (IsValidType(type_)) {
        t->set_type(type_);
      }
    }
  }

  void validate_views() {
    // DALI_ENFORCE(state_ != State::noncontiguous);
    // Return if we do not have a valid allocation
    if (!IsValidType(tl_->type()))
      return;
    if (!tl_->raw_data())
      return;

    for (size_t i = 0; i < tensors_.size(); i++) {
    views_count_ = tensors_.size();
      // TODO(klecki): deleter that reduces views_count? or just noop sharing?
      tensors_[i]->ShareData(
          std::shared_ptr<void>(tl_->raw_mutable_tensor(i),
                                [&views_count = views_count_](void *) { views_count--; }),
          volume(tl_->tensor_shape(i)), tl_->tensor_shape(i));
      // tensors_[i]->ShareData(tl_.get(), static_cast<int>(i));
    }
  }
  std::atomic<int> views_count_;
  std::vector<std::shared_ptr<Tensor<Backend>>> tensors_;
  std::shared_ptr<TensorList<Backend>> tl_;
  State state_ = State::contiguous;
  // pinned status and type info should be uniform
  bool pinned_ = true;
  TypeInfo type_ = TypeInfo();
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_VECTOR_H_
