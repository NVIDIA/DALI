// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/access_order.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/sample_view.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_list.h"


namespace dali {

/**
 * @brief Merges TensorList<Backend> and std::vector<std::shared_ptr<Tensor<Backend>>> APIs
 * providing an uniform way of handling a collection/batch of tensors_.
 *
 * Propagates Buffer calls to every tensor uniformly
 *
 * TODO(klecki): Expected improvements to TensorVector
 * 1. Remove superfluous indirection via shared_ptr to samples.
 * 2. Keep metadata (shape, sample_dim, layout, order) at batch level like we already do with type
 * 3. Detect and convert between contiguous and non-contiguous when possible:
 *    a. CopySample of bigger size
 *    b. Resize with coalesce option
 * 4. Contiguity check
 * @tparam Backend
 */
template <typename Backend>
class DLL_PUBLIC TensorVector {
 public:
  TensorVector();

  /**
   * @brief This constructor allows to create a TensorVector with `batch_size` samples,
   * that will be accessible as individual samples that can currently be individually resized which
   * is still utilized by the legacy operators.
   *
   * TODO(klecki): The API for empty tensor batch container of given number of samples
   * will be adjusted in next releases.
   */
  explicit TensorVector(int batch_size);

  explicit TensorVector(std::shared_ptr<TensorList<Backend>> tl);

  TensorVector(const TensorVector &) = delete;
  TensorVector &operator=(const TensorVector &) = delete;

  DLL_PUBLIC TensorVector<Backend>(TensorVector<Backend> &&other) noexcept;

  AccessOrder order() const {
    return order_;
  }

  /**
   * @brief Sets the associated access order.
   *
   * @note The caller must ensure that if `order` represents a CUDA stream, that stream
   *       is alive when this buffer is destroyed. This extends to buffers with which this
   *       one shares data. Use CUDAStreamPool::instance to get streams with indefinite lifetime.
   *
   * @param order       The new access order (stream or host). If the new order doesn't have
   *                    a value, the function has no effect.
   * @param synchronize If true, an appropriate synchronization is inserted between the old
   *                    and the new order. The caller may specify `false` if appropriate
   *                    synchronization is guaranteed by other means.
   */
  void set_order(AccessOrder order, bool synchronize = true);

  SampleView<Backend> operator[](size_t pos) {
    return {tensors_[pos]->raw_mutable_data(), tensors_[pos]->shape(), tensors_[pos]->type()};
  }

  ConstSampleView<Backend> operator[](size_t pos) const {
    return {tensors_[pos]->raw_data(), tensors_[pos]->shape(), tensors_[pos]->type()};
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

  size_t num_samples() const noexcept {
    return curr_tensors_size_;
  }

  int sample_dim() const {
    return IsContiguous() ? tl_->sample_dim() : num_samples() ? tensors_[0]->shape().size() : 0;
  }

  size_t total_nbytes() const noexcept;

  size_t total_capacity() const noexcept;

  std::vector<size_t> nbytes() const noexcept;

  std::vector<size_t> capacity() const noexcept;

  TensorListShape<> shape() const;

  const TensorShape<> &tensor_shape(int idx) const {
    return tensors_[idx]->shape();
  }

  /**
   * @brief Returns a typed pointer to the tensor with the given index.
   */
  template <typename T>
  DLL_PUBLIC inline T* mutable_tensor(int idx) {
    return tensors_[idx]->template mutable_data<T>();
  }

  /**
   * @brief Returns a const typed pointer to the tensor with the given index.
   */
  template <typename T>
  DLL_PUBLIC inline const T* tensor(int idx) const {
    return tensors_[idx]->template data<T>();
  }

  /**
   * @brief Returns a raw pointer to the tensor with the given index.
   */
  DLL_PUBLIC inline void* raw_mutable_tensor(int idx) {
    return tensors_[idx]->raw_mutable_data();
  }

  /**
   * @brief Returns a const raw pointer to the tensor with the given index.
   */
  DLL_PUBLIC inline const void* raw_tensor(int idx) const {
    return  tensors_[idx]->raw_data();
  }

  /**
   * @brief Analogue of TensorVector[dst].ShareData(owner[src]);
   *
   * The target TensorVector (this) must have enough samples for this to work (see SetSize()).
   * After this operation the TensorVector is converted into non-contiguous.
   *
   * Warning: If the TensorVector was contiguous, the samples that weren't overwritten by this
   * function would still report that they are sharing data. It is assumed that all samples are
   * replaced this way - TODO(klecki): this might be adjusted in follow-up.
   *
   * @param dst index of sample to be set
   * @param owner owner of source sample
   * @param src index of source sample in owner.
   */
  DLL_PUBLIC void UnsafeSetSample(int dst, const TensorVector<Backend> &owner, int src);

  /**
   * @brief Analogue of TensorVector[dst].ShareData(owner);
   *
   * The target TensorVector (this) must have enough samples for this to work (see SetSize()).
   * After this operation the TensorVector is converted into non-contiguous.
   *
   * Warning: If the TensorVector was contiguous, the samples that weren't overwritten by this
   * function would still report that they are sharing data. It is assumed that all samples are
   * replaced this way - TODO(klecki): this might be adjusted in follow-up.
   *
   * @param dst index of sample to be set
   * @param owner sample owner
   */
  DLL_PUBLIC void UnsafeSetSample(int dst, const Tensor<Backend> &owner);

  /**
   * @brief Analogue of TensorVector[dst].Copy(data[src]);
   *
   * The target TensorVector (this) must have enough samples for this to work (see SetSize()).
   * It must either be already non-contiguous or the shapes of copied samples must match exactly.
   *
   * Warning: It is assumed that the TensorVector is either first resized to desired shape,
   * or all samples are copied over. Automatically converting to non-contiguous container from
   * contiguous one by invoking copy of non-matching size is not supported yet.
   *
   * @param dst index of sample to be set
   * @param owner sample owner
   */
  DLL_PUBLIC void UnsafeCopySample(int dst, const TensorVector<Backend> &data, int src,
                                   AccessOrder order = {});

  DLL_PUBLIC void Resize(const TensorListShape<> &new_shape) {
    DALI_ENFORCE(IsValidType(type()),
                 "TensorVector has no type, 'set_type<T>()' or Resize(shape, type) must be called "
                 "on the TensorVector to set a valid type before it can be resized.");
    return Resize(new_shape, type());
  }

  DLL_PUBLIC void Resize(const TensorListShape<> &new_shape, DALIDataType new_type);

  /**
   * Change the number of tensors that can be accessed as samples without the need to
   * set them a size.
   * @param new_size
   */
  void SetSize(int new_size);

  void set_type(DALIDataType new_type);

  template <typename T>
  void set_type() {
    set_type(TypeTable::GetTypeId<T>());
  }

  DALIDataType type() const;

  const TypeInfo &type_info() const;

  /** @brief Set uniform layout for all samples in the list */
  void SetLayout(const TensorLayout &layout);

  TensorLayout GetLayout() const;

  DALIMeta &GetMeta(int idx);
  const DALIMeta &GetMeta(int idx) const;

  void SetMeta(int idx, const DALIMeta &meta);

  void set_pinned(bool pinned);

  bool is_pinned() const;

  int device_id() const;

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
  void Copy(const TensorList<SrcBackend> &in_tl, AccessOrder order = {});

  template <typename SrcBackend>
  void Copy(const TensorVector<SrcBackend> &in_tv, AccessOrder order = {});

  void ShareData(const TensorList<Backend> &in_tl);

  void ShareData(const TensorVector<Backend> &tv);

  TensorVector<Backend> &operator=(TensorVector<Backend> &&other) noexcept;

  void UpdateViews();

  shared_ptr<TensorList<Backend>> AsTensorList(bool check_contiguity = true);

 private:
  enum class State { contiguous, noncontiguous };

  // Forward declarations in signature, beware
  friend void MakeSampleView(class SampleWorkspace &sample, class HostWorkspace &batch,
                             int data_idx, int thread_idx);
  friend void EnforceCorrectness(class HostWorkspace &ws, bool contiguous);

  /**
   * @brief After RunImpl(SampleWorkspace&) operated on individual samples without propagating
   * the allocation metadata back to the the batch structure, take that metadata from the samples
   * and update it in TensorVector.
   *
   * @param contiguous if the Tensor was previously preallocated and should remain contiguous
   * or be treated as non-contiguous set of individual samples.
   */
  void PropagateUp(bool contiguous);

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
  TypeInfo type_{};
  AccessOrder order_;

  // So we can access the members of other TensorVectors
  // with different template types
  template <typename InBackend>
  friend class TensorVector;
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_VECTOR_H_
