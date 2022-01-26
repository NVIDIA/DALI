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
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_list.h"

#include "dali/core/tensor_shape.h"

#include "dali/core/tensor_view.h"
#include "dali/core/backend_tags.h"


namespace dali {


/**
 * @brief Maps DALI Backend to dali::kernels storage backend.
 */
template <typename Backend>
struct storage_tag_map3;

template <>
struct storage_tag_map3<CPUBackend> {
  using type = StorageCPU;
};

template <>
struct storage_tag_map3<GPUBackend> {
  using type = StorageGPU;
};

template <typename Backend>
using storage_tag_map3_t = typename storage_tag_map3<Backend>::type;
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
    return tl_->order();
  }

  void set_order(AccessOrder order, bool synchronize = true);

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

  // DLL_PUBLIC void SetSample(int idx, const Buffer<Backend> &owner) {
  //   Tensor<Backend> tmp;
  //   tmp.ShareData(owner);
  //   tmp.Resize(shape()[idx], type());
  //   SetSample(idx, owner);
  //   // TODO checks
  //   // todo share_data replacement
  //   // tensor_[idx].ShareData(owner);
  // }

  DLL_PUBLIC void SetSample(int idx, const Tensor<Backend> &owner) {
    // TODO checks
    // DALI_ENFORCE(owner.shape().sample_dim() == shape().sample_dim(), "Sample must have the same dim");
    if (type() == DALI_NO_TYPE && owner.type() != DALI_NO_TYPE) {
      set_type(owner.type());
    }
    DALI_ENFORCE(type() == owner.type(), "Sample must have the same type as batch");
    // kind (pinned?), order, layout, etc...
    // The metadata

    if (tensors_[idx]->shape().num_elements() != owner.shape().num_elements()) {
      SetContiguous(false);
    }
    tensors_[idx]->ShareData(owner);
    // todo v update shape
    // shape().set_tensor_shape(idx, owner.shape());
  }

  DLL_PUBLIC void CopySample(int dst, const TensorVector<Backend> &data, int src, AccessOrder order = {}) {
    // TODO checks
    // DALI_ENFORCE(owner.shape().sample_dim() == shape().sample_dim(), "Sample must have the same dim");
    if (type() == DALI_NO_TYPE && data.type() != DALI_NO_TYPE) {
      set_type(data.type());
    }
    DALI_ENFORCE(type() == data.type(), "Sample must have the same type as batch");
    // kind (pinned?), order, layout, etc...
    // The metadata

    if (tensors_[dst]->shape().num_elements() != data.tensors_[src]->shape().num_elements()) {
      SetContiguous(false);
    }
    tensors_[dst]->Copy(*(data.tensors_[src]), order);
    // todo v update shape
    // shape().set_tensor_shape(idx, owner.shape());
  }

  DLL_PUBLIC TensorView<storage_tag_map3_t<Backend>, void, DynamicDimensions> operator[](
      int sample_idx) {
    return {tensors_[sample_idx]->raw_mutable_data(), tensor_shape(sample_idx), type()};
  }

  DLL_PUBLIC TensorView<storage_tag_map3_t<Backend>, const void, DynamicDimensions> operator[](
      int sample_idx) const {
    return {tensors_[sample_idx]->raw_data(), tensor_shape(sample_idx), type()};
  }


  Tensor<Backend> &GetSample(size_t pos) {
    return *(tensors_[pos]);
  }

  const Tensor<Backend> &GetSample(size_t pos) const {
    return *(tensors_[pos]);
  }

  // Tensor<Backend> &operator[](size_t pos) {
  //   return *(tensors_[pos]);
  // }

  // const Tensor<Backend> &operator[](size_t pos) const {
  //   return *(tensors_[pos]);
  // }

  auto tensor_handle(size_t pos) {
    return tensors_[pos];
  }

  auto tensor_handle(size_t pos) const {
    return tensors_[pos];
  }

  // auto begin() noexcept {
  //   return tensors_.begin();
  // }

  // auto begin() const noexcept {
  //   return tensors_.begin();
  // }

  // auto cbegin() const noexcept {
  //   return tensors_.cbegin();
  // }

  // auto end() noexcept {
  //   return tensors_.end();
  // }

  // auto end() const noexcept {
  //   return tensors_.end();
  // }

  // auto cend() const noexcept {
  //   return tensors_.cend();
  // }

  size_t num_samples() const noexcept {
    return curr_tensors_size_;
  }

  int sample_dim() const {
    return IsContiguous() ? tl_->sample_dim() : num_samples() ? tensors_[0]->shape().size() : 0;
  }

  size_t nbytes() const noexcept;

  size_t capacity() const noexcept;

  TensorListShape<> shape() const;

  const TensorShape<> &tensor_shape(int idx) const {
    return tensors_[idx]->shape();
  }

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

  int device_id() const {
    return 0;  // TODO fixme
  }

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

  // So we can access the members of other TensorVectors
  // with different template types
  template <typename InBackend>
  friend class TensorVector;
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_VECTOR_H_
