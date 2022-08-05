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
#include <string>
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

  int num_samples() const noexcept {
    return curr_num_tensors_;
  }

  void set_sample_dim(int sample_dim);

  int sample_dim() const {
    return sample_dim_;
  }

  size_t nbytes() const noexcept;

  size_t capacity() const noexcept;

  /**
   * @brief Returns the size in bytes of the underlying data chunks
   * TODO(klecki): Temporary API to be reworked, do not use.
   */
  std::vector<size_t> _chunks_nbytes() const;

  /**
   * @brief Returns the real size of the underlying allocations
   * TODO(klecki): Temporary API to be reworked, do not use.
   */
  std::vector<size_t> _chunks_capacity() const;

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
   * @brief Analogue of TensorVector[sample_idx].ShareData(src[src_sample_idx]);
   *
   * The target TensorVector (this) must have enough samples for this to work (see SetSize()).
   * After this operation the TensorVector is converted into non-contiguous.
   *
   * Warning: If the TensorVector was contiguous, the samples that weren't overwritten by this
   * function would still report that they are sharing data. It is assumed that all samples are
   * replaced this way - TODO(klecki): this might be adjusted in follow-up.
   *
   * @param sample_idx index of sample to be set
   * @param src owner of source sample
   * @param src_sample_idx index of source sample in owner.
   */
  DLL_PUBLIC void UnsafeSetSample(int sample_idx, const TensorVector<Backend> &src,
                                  int src_sample_idx);

  /**
   * @brief Analogue of TensorVector[sample_idx].ShareData(owner);
   *
   * The target TensorVector (this) must have enough samples for this to work (see SetSize()).
   * After this operation the TensorVector is converted into non-contiguous.
   *
   * Warning: If the TensorVector was contiguous, the samples that weren't overwritten by this
   * function would still report that they are sharing data. It is assumed that all samples are
   * replaced this way - TODO(klecki): this might be adjusted in follow-up.
   *
   * @param sample_idx index of sample to be set
   * @param src sample owner
   */
  DLL_PUBLIC void UnsafeSetSample(int sample_idx, const Tensor<Backend> &src);

  /**
   * @brief Analogue of TensorVector[sample_idx].ShareData for externally provided memory.
   *
   * The target TensorVector (this) must have enough samples for this to work (see SetSize()).
   * After this operation the TensorVector is converted into non-contiguous.
   *
   * Warning: If the TensorVector was contiguous, the samples that weren't overwritten by this
   * function would still report that they are sharing data. It is assumed that all samples are
   * replaced this way - TODO(klecki): this might be adjusted in follow-up.
   *
   * The metadata (pinned, type, device_id, order, layout) must match what is already set
   * for the whole batch to maintain consistency.
   */
  DLL_PUBLIC void UnsafeSetSample(int sample_idx, const shared_ptr<void> &ptr, size_t bytes,
                                  bool pinned, const TensorShape<> &shape, DALIDataType type,
                                  int device_id, AccessOrder order = {},
                                  const TensorLayout &layout = "");

  /**
   * @brief Analogue of TensorVector[sample_idx].Copy(src[src_sample_idx]);
   *
   * The target TensorVector (this) must have enough samples for this to work (see SetSize()).
   * It must either be already non-contiguous or the shapes of copied samples must match exactly.
   *
   * Warning: It is assumed that the TensorVector is either first resized to desired shape,
   * or all samples are copied over. Automatically converting to non-contiguous container from
   * contiguous one by invoking copy of non-matching size is not supported yet.
   *
   * @param sample_idx index of sample to be set
   * @param src sample owner
   * @param src_sample_idx index of source sample in owner.
   */
  DLL_PUBLIC void UnsafeCopySample(int sample_idx, const TensorVector<Backend> &src,
                                   int src_sample_idx, AccessOrder order = {});

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

  /**
   * @name Configuration cloning
   * @{
   */
  /**
   * @brief Setup all the batch properties of this TensorVector the same way as the provided tensor
   * or batch.
   *
   * Precondition: the TensorVector should not have data.
   *
   * Configures: type, layout, pinned, order and dimensionality.
   */
  void SetupLike(const Tensor<Backend> &sample) {
    SetupLikeImpl(sample);
  }

  void SetupLike(const TensorVector<Backend> &other) {
    SetupLikeImpl(other);
  }

  void SetupLike(const TensorList<Backend> &other) {
    SetupLikeImpl(other);
  }
  /** @} */

  /**
   * @name Type setting functions.
   * @{
   */
  /**
   * @brief Set the type of the current batch. The type needs to be set before calling
   * the Resize(const TensorListShape<> &) function. It cannot be used to change the type after
   * allocation happened.
   *
   * Resize(const TensorListShape<> &, DALIDataType) can be used without prior set_type call or to
   * request a different type after allocation.
   */
  void set_type(DALIDataType new_type);

  template <typename T>
  void set_type() {
    set_type(TypeTable::GetTypeId<T>());
  }
  /** @} */


  DALIDataType type() const;

  const TypeInfo &type_info() const;

  /** @brief Set uniform layout for all samples in the list */
  void SetLayout(const TensorLayout &layout);

  void SetSkipSample(int idx, bool skip_sample);

  void SetSourceInfo(int idx, const std::string& source_info);

  TensorLayout GetLayout() const;

  const DALIMeta &GetMeta(int idx) const;

  void SetMeta(int idx, const DALIMeta &meta);

  void set_pinned(bool pinned);

  bool is_pinned() const;

  void set_device_id(int device_id);

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

  /**
   * @brief Checks whether the batch container is contiguous. It returns true if and only if
   * all of the stored individual tensors are densely packed in memory.
   */
  bool IsContiguousInMemory() const;

  /**
   * @brief Checks whether the batch container can be converted to a dense Tensor. It returns true
   * if and only if all of the stored tensors have the same shape and they are densely packed in
   * memory.
   */
  bool IsDenseTensor() const;

  /**
   * @brief Returns a Tensor which shares the data with this batch object and give it the
   * provided shape. Batch and the Tensor share the memory allocation. The tensor obtained through
   * this function stays valid for as long as TensorList data is unchanged.
   * The batch must be representable as DenseTensor.
   */
  DLL_PUBLIC Tensor<Backend> AsReshapedTensor(const TensorShape<> &new_shape);

  /**
   * @brief Return a Dense Tensor representation of the underlying memory if possible.
   */
  DLL_PUBLIC Tensor<Backend> AsTensor();

 private:
  enum class State { contiguous, noncontiguous };

  // Forward declarations in signature, beware
  friend void MakeSampleView(class SampleWorkspace &sample, class HostWorkspace &batch,
                             int data_idx, int thread_idx);
  friend void FixBatchPropertiesConsistency(class HostWorkspace &ws, bool contiguous);

  auto tensor_handle(size_t pos) {
    return tensors_[pos];
  }

  auto tensor_handle(size_t pos) const {
    return tensors_[pos];
  }

  template <typename T>
  void SetupLikeImpl(const T &other) {
    DALI_ENFORCE(!has_data(),
                "Batch object can be initialized this way only when it isn't allocated.");
    set_type(other.type());
    set_sample_dim(other.shape().sample_dim());
    SetLayout(other.GetLayout());
    set_order(other.order());
    set_pinned(other.is_pinned());
  }

  /**
   * @brief After RunImpl(SampleWorkspace&) operated on individual samples without propagating
   * the allocation metadata back to the the batch structure, take that metadata from the samples
   * and update it in TensorVector.
   *
   * @param contiguous if the Tensor was previously preallocated and should remain contiguous
   * or be treated as non-contiguous set of individual samples.
   */
  void UpdatePropertiesFromSamples(bool contiguous);

  bool has_data() const;

  void resize_tensors(int size);

  void update_view(int idx);

  std::vector<std::shared_ptr<Tensor<Backend>>> tensors_;
  int curr_num_tensors_;
  std::shared_ptr<TensorList<Backend>> tl_;
  State state_ = State::noncontiguous;
  // pinned status and type info should be uniform
  bool pinned_ = true;
  TypeInfo type_{};
  int sample_dim_ = -1;
  AccessOrder order_;

  // So we can access the members of other TensorVectors
  // with different template types
  template <typename InBackend>
  friend class TensorVector;

  /** @defgroup AccessorFunctions Fallback for accessing pointers owning the samples
   * Fallback access to contiguous data or samples of the batch. It should not be used for regular
   * processing, intended mostly for batches that were made sure to be contiguous (mainly
   * for pipeline outputs).
   * @{
   */

  /**
   * @brief Return an un-typed pointer to the underlying storage.
   * The TensorVector must be either empty or have a valid type and be contiguous.
   */
  friend void *unsafe_raw_mutable_data(TensorVector<Backend> &tv) {
    DALI_ENFORCE(tv.IsContiguous(), "Data pointer can be obtain only for contiguous TensorVector.");
    return unsafe_raw_mutable_data(*tv.tl_);
  }

  /**
   * @brief Return an un-typed const pointer to the underlying storage.
   * The TensorVector must be either empty or have a valid type and be contiguous.
   */
  friend const void *unsafe_raw_data(const TensorVector<Backend> &tv) {
    DALI_ENFORCE(tv.IsContiguous(), "Data pointer can be obtain only for contiguous TensorVector.");
    return unsafe_raw_data(*tv.tl_);
  }


  /**
   * @brief Return the shared pointer, that we can use to correctly share the ownership of sample
   * with.
   * Sample 0 is aliased with the whole buffer, if it is contiguous.
   */
  friend shared_ptr<void> unsafe_sample_owner(TensorVector<Backend> &batch, int sample_idx) {
    // create new aliasing pointer to current data allocation, so we share the use count
    // and the deleter correctly.
    if (batch.IsContiguous()) {
      return {unsafe_sample_owner(*batch.tl_, 0), batch.raw_mutable_tensor(sample_idx)};
    } else {
      return batch.tensors_[sample_idx]->get_data_ptr();
    }
  }

  /**
   * @brief Return the shared pointer, that we can use to correctly share the ownership of batch
   * with.
   * Only allowed for contiguous batch, in typical scenario it is equivalent to
   * unsafe_sample_owner(batch, 0)
   */
  friend shared_ptr<void> unsafe_owner(TensorVector<Backend> &batch) {
    DALI_ENFORCE(batch.IsContiguous(),
                 "Data owner pointer can be obtain only for contiguous TensorVector.");
    return unsafe_owner(*batch.tl_);
  }

  /** @} */  // end of AccessorFunctions
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_VECTOR_H_
