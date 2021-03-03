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
#include <cassert>
#include <memory>
#include <utility>
#include <vector>

#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_list.h"

#include "dali/core/tensor_shape.h"

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
class DLL_PUBLIC TensorVector {
 public:
  TensorVector();

  explicit TensorVector(int batch_size);

  explicit TensorVector(std::shared_ptr<TensorList<Backend>> tl);

  TensorVector(const TensorVector &) = delete;
  TensorVector &operator=(const TensorVector &) = delete;

  DLL_PUBLIC TensorVector<Backend>(TensorVector<Backend> &&other) noexcept;

  Tensor<Backend> &operator[](size_t pos) {
    return *(tensors_[pos]);
  }

  const Tensor<Backend> &operator[](size_t pos) const {
    return *(tensors_[pos]);
  }

  auto tensor_handle(size_t pos) {
    return tensors_[pos];
  }

  auto tensor_handle(size_t pos) const {
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
    return curr_tensors_size_;
  }

  size_t ntensor() const noexcept {
    return curr_tensors_size_;
  }

  int sample_dim() const {
    return IsContiguous() ? tl_->sample_dim() : ntensor() ? tensors_[0]->shape().size() : 0;
  }

  size_t nbytes() const noexcept;

  size_t capacity() const noexcept;

  TensorListShape<> shape() const;

  const TensorShape<> &tensor_shape(int idx) const {
    return tensors_[idx]->shape();
  }

  const void *raw_tensor(int idx) const {
    return tensors_[idx]->raw_data();
  }

  void* raw_mutable_tensor(int idx) {
    return tensors_[idx]->raw_mutable_data();
  }

  DLL_PUBLIC void Resize(const TensorListShape<> &new_shape) {
    return Resize(new_shape, type());
  }

  DLL_PUBLIC void Resize(const TensorListShape<> &new_shape, const TypeInfo &new_type);

  /**
   * Change the number of tensors in the TensorVector, without the need of
   * specifying the shape of every such tensor. When setting the new size,
   * this function will retain the shapes that already exist. New tensors
   * are given a shape of 0-volume and appropriate dimension.
   * @param new_size
   */
  void SetSize(int new_size);

  void set_type(const TypeInfo &new_type);

  const TypeInfo &type() const;

  /** @brief Set uniform layout for all samples in the list */
  void SetLayout(const TensorLayout &layout);

  TensorLayout GetLayout() const;

  const DALIMeta &GetMeta(int idx) const;

  void SetMeta(int idx, const DALIMeta &meta);

  void set_pinned(bool pinned);

  bool is_pinned() const;

  /**
   * @brief Reserve as contiguous tensor list internally
   */
  void reserve(size_t total_bytes);

  /**
   * @brief Reserve as vector of `batch_size` tensors internally
   */
  void reserve(size_t bytes_per_sample, int batch_size);

  /**
   * @brief If the TensorVector is backed by TensorList (contiguous memory)
   */
  bool IsContiguous() const noexcept;

  /**
   * @brief Set the current state if further calls like Resize() or set_type
   *        should use TensorList or std::vector<Tensor> as backing memory
   */
  void SetContiguous(bool contiguous);

  void Reset();

  template <typename SrcBackend>
  void Copy(const TensorList<SrcBackend> &in_tl, cudaStream_t stream);

  template <typename SrcBackend>
  void Copy(const TensorVector<SrcBackend> &in_tv, cudaStream_t stream);

  void ShareData(TensorList<Backend> *in_tl);

  void ShareWith(TensorList<Backend> *in_tl) const;

  void ShareData(TensorVector<Backend> *tv);

  TensorVector<Backend> &operator=(TensorVector<Backend> &&other) noexcept;

  void UpdateViews();

  shared_ptr<TensorList<Backend>> AsTensorList(bool check_contiguity = true);

 private:
  enum class State { contiguous, noncontiguous };

  struct ViewRefDeleter {
    void operator()(void*) { --*ref; }
    std::atomic<int> *ref;
  };

  void resize_tensors(int size);

  void update_view(int idx);

  std::atomic<int> views_count_;
  std::vector<std::shared_ptr<Tensor<Backend>>> tensors_;
  size_t curr_tensors_size_;
  std::shared_ptr<TensorList<Backend>> tl_;
  State state_ = State::noncontiguous;
  // pinned status and type info should be uniform
  bool pinned_ = true;
  TypeInfo type_ = TypeInfo();

  // So we can access the members of other TensorVectors
  // with different template types
  template <typename InBackend>
  friend class TensorVector;
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_VECTOR_H_
