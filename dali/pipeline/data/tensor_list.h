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

#ifndef DALI_PIPELINE_DATA_TENSOR_LIST_H_
#define DALI_PIPELINE_DATA_TENSOR_LIST_H_

#include <atomic>
#include <cassert>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/access_order.h"
#include "dali/core/error_handling.h"
#include "dali/core/tensor_layout.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/sample_view.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/types.h"


namespace dali {

// Forward declarations in signature, beware
namespace test {
class TensorListVariableBatchSizeTest_UpdatePropertiesFromSamples_Test;
};  // namespace test

template <typename Backend>
class DLL_PUBLIC TensorList;

/**
 * @brief Size of stack-based array used to prepare the pointers and other parameters for
 * operations on batch, like TypeInfo::Copy.
 * For bigger batches, the list of pointers/sizes would be stored as dynamic allocation
 * (SmallVector), used in a common pattern where we have a copy from or to a batch of samples.
 */
constexpr size_t kMaxStaticCopyBatchSize = 256;

/**
 * @brief SmallVector alias used when dealing with batches of data in common operations
 *
 * The static stack allocation size is adjusted for that purpose.
 */
template <typename T>
using BatchVector = SmallVector<T, kMaxStaticCopyBatchSize>;


/**
 * @brief Data structure representing a batch of non-uniformly shaped Tensor.
 * Type, dimensionality, order, pinned status and layout are uniform for all samples.
 *
 * TODO(klecki): Additional followups for TensorList:
 *   * Based on intended usage patterns extend CopySample to switch the batch into noncontiguous
 *     mode and/or introduce ResizeSample functionality.
 * @tparam Backend
 */
template <typename Backend>
class DLL_PUBLIC TensorList {
 public:
  TensorList();

  /**
   * @brief This constructor allows to create a TensorList with `batch_size` samples.
   * Automatically sets dimension to 1, and the sample shape is {0}.
   */
  explicit TensorList(int batch_size);

  TensorList(const TensorList &) = delete;
  TensorList &operator=(const TensorList &) = delete;

  TensorList<Backend> &operator=(TensorList<Backend> &&other) noexcept;
  DLL_PUBLIC TensorList<Backend>(TensorList<Backend> &&other) noexcept;


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

  /**
   * @name Shape access
   * @{
   */
  /**
   * @brief Get the number of samples this batch holds.
   */
  int num_samples() const noexcept {
    return curr_num_tensors_;
  }

  /**
   * @brief Number of elements in batch (total of all samples).
   *
   * Note: The usage of this member function is intended to be reworked in following updates.
   * For this purpose the name is distinct so we can easily search and replace.
   * [shape_access]
   */
  int64_t _num_elements() const {
    return shape().num_elements();
  }

  /**
   * @brief Set the dimensionality for all samples in the batch
   */
  void set_sample_dim(int sample_dim);

  /**
   * @brief Get the dimensionality of the sample in the batch
   */
  int sample_dim() const {
    return sample_dim_;
  }

  /**
   * @brief Get the shape of the batch.
   */
  const TensorListShape<> &shape() const &;

  /**
   * @brief Get the shape of the sample.
   *
   * [shape_access]
   */
  const TensorShape<> &tensor_shape(int idx) const & {
    return tensors_[idx].shape();
  }

  /**
   * @brief Get the shape of the sample.
   *
   * [shape_access]
   */
  inline span<const int64_t> tensor_shape_span(int idx) const & {
    return shape_.tensor_shape_span(idx);
  }
  /** @} */


  /**
   * @name Sample access
   * @{
   */
  /**
   * @brief Get the view for the sample at given position
   */
  SampleView<Backend> operator[](size_t pos);

  /**
   * @brief Get the view for the sample at given position
   */
  ConstSampleView<Backend> operator[](size_t pos) const;

  /**
   * @brief Returns a typed pointer to the tensor with the given index.
   */
  template <typename T>
  DLL_PUBLIC inline T *mutable_tensor(int idx) {
    return tensors_[idx].template mutable_data<T>();
  }

  /**
   * @brief Returns a const typed pointer to the tensor with the given index.
   */
  template <typename T>
  DLL_PUBLIC inline const T *tensor(int idx) const {
    return tensors_[idx].template data<T>();
  }

  /**
   * @brief Returns a raw pointer to the tensor with the given index.
   */
  DLL_PUBLIC inline void *raw_mutable_tensor(int idx) {
    return tensors_[idx].raw_mutable_data();
  }

  /**
   * @brief Returns a const raw pointer to the tensor with the given index.
   */
  DLL_PUBLIC inline const void *raw_tensor(int idx) const {
    return tensors_[idx].raw_data();
  }
  /** @} */

  /**
   * @name Sample setting (sharing)
   * @{
   */
  /**
   * @brief Analogue of TensorList[sample_idx].ShareData(src[src_sample_idx]);
   *
   * The target TensorList (this) must have enough samples for this to work (see SetSize()).
   * After this operation the TensorList is converted into non-contiguous.
   *
   * Warning: If the TensorList was contiguous, the samples that weren't overwritten by this
   * function would still report that they are sharing data. It is advised that all samples are
   * replaced this way otherwise the contiguous allocation would be kept alive.
   *
   * The metadata (pinned, type, device_id, layout) must match what is already set for the whole
   * batch to maintain consistency.
   *
   * We wait for the order of incoming sample in the order of the batch to allow correctly ordered
   * access of the new sample.
   *
   * @param sample_idx index of sample to be set
   * @param src owner of source sample
   * @param src_sample_idx index of source sample in owner.
   */
  DLL_PUBLIC void SetSample(int sample_idx, const TensorList<Backend> &src, int src_sample_idx);

  /**
   * @brief Analogue of TensorList[sample_idx].ShareData(owner);
   *
   * The target TensorList (this) must have enough samples for this to work (see SetSize()).
   * After this operation the TensorList is converted into non-contiguous.
   *
   * Warning: If the TensorList was contiguous, the samples that weren't overwritten by this
   * function would still report that they are sharing data. It is advised that all samples are
   * replaced this way otherwise the contiguous allocation would be kept alive.
   *
   * The metadata (pinned, type, device_id, layout) must match what is already set for the whole
   * batch to maintain consistency.
   *
   * We wait for the order of incoming sample in the order of the batch to allow correctly ordered
   * access of the new sample.
   *
   * @param sample_idx index of sample to be set
   * @param src sample owner
   */
  DLL_PUBLIC void SetSample(int sample_idx, const Tensor<Backend> &src);

  /**
   * @brief Analogue of TensorList[sample_idx].ShareData for externally provided memory.
   *
   * The target TensorList (this) must have enough samples for this to work (see SetSize()).
   * After this operation the TensorList is converted into non-contiguous.
   *
   * Warning: If the TensorList was contiguous, the samples that weren't overwritten by this
   * function would still report that they are sharing data. It is advised that all samples are
   * replaced this way otherwise the contiguous allocation would be kept alive.
   *
   * The metadata (pinned, type, device_id, layout) must match what is already set for the whole
   * batch to maintain consistency.
   *
   * We wait for the order of incoming sample in the order of the batch to allow correctly ordered
   * access of the new sample.
   */
  DLL_PUBLIC void SetSample(int sample_idx, const shared_ptr<void> &ptr, size_t bytes, bool pinned,
                            const TensorShape<> &shape, DALIDataType type, int device_id,
                            AccessOrder order, const TensorLayout &layout = "");
  /** @} */

  /**
   * @name Sample copying
   * @{
   */
  /**
   * @brief Analogue of TensorList[sample_idx].Copy(src[src_sample_idx]);
   *
   * The target TensorList (this) must have enough samples for this to work (see SetSize()).
   * It must either be already non-contiguous or the shapes of copied samples must match exactly.
   *
   * Warning: It is assumed that the TensorList is either first resized to desired shape,
   * or all samples are copied over. Automatically converting to non-contiguous container from
   * contiguous one by invoking copy of non-matching size is not supported yet.
   *
   * @param sample_idx index of sample to be set
   * @param src sample owner
   * @param src_sample_idx index of source sample in owner.
   */
  DLL_PUBLIC void CopySample(int sample_idx, const TensorList<Backend> &src, int src_sample_idx,
                             AccessOrder order = {});

  DLL_PUBLIC void CopySample(int sample_idx, const Tensor<Backend> &src, AccessOrder order = {});
  /** @} */


  /**
   * @name Type access
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

  /**
   * @brief Get the type of samples in the batch.
   */
  DALIDataType type() const;

  /**
   * @brief Get the TypeInfo of samples in the batch.
   *
   * @note Using DALIDataType via type() is recommended over accessing type_info().
   */
  const TypeInfo &type_info() const;
  /** @} */

  /**
   * @name Size and shape, and allocation changing
   * @{
   */
  /**
   * Change the number of samples in the batch without specifying their shape or type (that is
   * calling the Resize).
   * It can be used to adjust the batch to given size and than to set individual samples.
   * If the new_size is bigger than the current one, new samples have 0-element shape of
   * correct sample dimensionality.
   */
  void SetSize(int new_size);

  /**
   * @brief Resize the batch to fit the new shape
   * See Resize(const TensorListShape<> &, DALIDataType, BatchContiguity) for details
   */
  DLL_PUBLIC void Resize(const TensorListShape<> &new_shape) {
    DALI_ENFORCE(IsValidType(type()),
                 "TensorList has no type, 'set_type<T>()' or Resize(shape, type) must be called "
                 "on the TensorList to set a valid type before it can be resized.");
    return Resize(new_shape, type());
  }

  /**
   * @brief Resize the batch to fit the new shape. It is possible to change the type and
   * dimensionality this way, as well as specify if we want the allocation to happen for individual
   * samples or contiguous one for all samples. If the currently allocated memory is enough for the
   * requested shape, no allocation would be made. In non-contiguous mode (or when switching to it),
   * each sample would be resized individually.
   *
   * Resizing samples that are using external backing allocation (sharing data via SetSample)
   * is not allowed to cause new allocation.
   * @param new_shape requested shape
   * @param new_type requested type
   * @param state Optional change of contiguity mode.
   *    * Automatic keeps the current one or use one allocation if reallocation is needed
   *    * Contiguous forces the allocation to be contiguous
   *    * Noncontiguous - detach all samples, and use them separately, the contiguous buffer
   *      might still be used as backing storage until new allocations are needed for all samples
   */
  DLL_PUBLIC void Resize(const TensorListShape<> &new_shape, DALIDataType new_type,
                         BatchContiguity state = BatchContiguity::Automatic);

  /**
   * @brief Resize individual sample. Allowed only in non-contiguous mode - it will convert the
   * TensorList on the first call. The type must be already known, and the TensorList must heave
   * enough elements for this operation.
   *
   * @param sample_idx sample index to be resized
   * @param new_shape requested shape
   */
  DLL_PUBLIC void ResizeSample(int sample_idx, const TensorShape<> &new_shape);

  /**
   * @brief Reserve memory as one contiguous allocation
   */
  void reserve(size_t total_bytes);

  /**
   * @brief Reserve as vector of `batch_size` allocations internally
   */
  void reserve(size_t bytes_per_sample, int batch_size);
  /** @} */

  /**
   * @name Configuration cloning
   * @{
   */
  /**
   * @brief Setup all the batch properties of this TensorList the same way as the provided tensor
   * or batch.
   *
   * Precondition: the TensorList should not have data.
   *
   * Configures: type, layout, pinned, order and dimensionality.
   */
  void SetupLike(const Tensor<Backend> &sample) {
    SetupLikeImpl(sample);
  }

  void SetupLike(const TensorList<Backend> &other) {
    SetupLikeImpl(other);
  }
  /** @} */

  /**
   * @name Configure contiguity of allocations
   * @{
   */
  /**
   * @brief If the batch is backed by contiguous buffer
   */
  bool IsContiguous() const noexcept;

  /**
   * @brief Pin the current state for further allocating calls like Resize() or set_type
   *        to use contiguous or noncontiguous backing memory.
   *        Setting BatchContiguity::Automatic allows to change it with every call to Resize().
   */
  void SetContiguity(BatchContiguity state);

  /**
   * @brief Check the batch contiguity state.
   */
  BatchContiguity GetContiguity() const noexcept;

  /**
   * @brief Coalesce from individual samples to a contiguous buffer if the conditions are met.
   * TODO(klecki): NOT YET IMPLEMENTED.
   */
  void MakeContiguous(std::weak_ptr<void> owner = {});

  /**
   * @brief Transform from contiguous allocation to individual samples without adjusting
   * the allocation.
   */
  void MakeNoncontiguous();
  /** @} */

  /**
   * @brief Reset the allocations and most properties.
   * Device related properties (device id, memory pinning, order) and contiguity status are not
   * reset.
   */
  void Reset();

  /**
   * @brief Copy whole batch
   */
  template <typename SrcBackend>
  void Copy(const TensorList<SrcBackend> &in_tv, AccessOrder order = {},
            bool use_copy_kernel = false);

  /**
   * @brief Set the provided buffer as backing memory for this batch.
   */
  DLL_PUBLIC void ShareData(const shared_ptr<void> &ptr, size_t bytes, bool pinned,
                            const TensorListShape<> &shape, DALIDataType type, int device_id,
                            AccessOrder order = {}, const TensorLayout &layout = "");

  /**
   * @brief Set other batch as backing memory for this one. Preserves the contiguity status.
   */
  void ShareData(const TensorList<Backend> &tv);

  void set_pinned(bool pinned);

  bool is_pinned() const;

  void set_device_id(int device_id);

  int device_id() const;

  bool has_data() const;

  bool shares_data() const;

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

  /**
   * @name Metadata access
   * @{
   */
  /**
   * @brief Set uniform layout for all samples in the batch
   */
  void SetLayout(const TensorLayout &layout);

  /**
   * @brief Get the layout of the sample in the batch.
   */
  TensorLayout GetLayout() const;

  /**
   * @brief Set cache metadata for given sample
   */
  void SetSkipSample(int idx, bool skip_sample);

  /**
   * @brief Set source information for given sample
   */
  void SetSourceInfo(int idx, const std::string &source_info);

  /**
   * @brief Get the metadata for given sample
   */
  const DALIMeta &GetMeta(int idx) const;

  /**
   * @brief Set the metadata for given sample
   */
  void SetMeta(int idx, const DALIMeta &meta);
  /** @} */

  /**
   * @name Allocation metadata
   * @{
   */
  /**
   * @brief Returns the total size in bytes of the underlying data chunks.
   * @note Keep in mind, that the memory can be fragmented.
   */
  size_t nbytes() const noexcept;

  /**
   * @brief Returns the sum of all underlying allocations.
   * @note Keep in mind, that the memory can be fragmented.
   */
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
  /** @} */

 private:
  /**
   * @brief Tracking the contiguous/noncontiguous state of the batch.
   * By default we keep what was previously set when resizing and can change it during Resize
   * unless it is enforced.
   */
  class State {
   public:
    // TODO(klecki): Any sensible defaults?
    State() : contiguous_(false), forced_(false) {}
    State(BatchContiguity state, bool forced) {
      Setup(state, forced);
    }
    State(const State &) = default;
    State &operator=(const State &) = default;

    /**
     * @brief Override current state.
     */
    void Setup(BatchContiguity state, bool forced = false) {
      if (forced) {
        DALI_ENFORCE(
            state == BatchContiguity::Contiguous || state == BatchContiguity::Noncontiguous,
            "Only specific state can be enforced");
      }
      if (state != BatchContiguity::Automatic) {
        contiguous_ = state == BatchContiguity::Contiguous;
      }
      forced_ = forced;
    }

    /**
     * @brief Update current state obeying the enforced state.
     * BatchContiguity::Automatic is always allowed and does not change the state
     *
     * State can be changed unless it is enforced, in that case it will raise an error.
     *
     * @return true if the state changed.
     */
    bool Update(BatchContiguity requested_state) {
      if (requested_state == BatchContiguity::Automatic) {
        return false;
      }
      if (forced_) {
        DALI_ENFORCE(Get() == requested_state,
                     make_string("State cannot be changed as it is enforced to ",
                                 contiguous_ ? "contiguous." : "noncontiguous."));
      }
      if (Get() == requested_state) {
        return false;
      }
      // Here it is guaranteed that we are changing state, so swap it.
      contiguous_ = !contiguous_;
      return true;
    }

    bool IsContiguous() const {
      return contiguous_;
    }

    bool IsForced() const {
      return forced_;
    }

    BatchContiguity Get() const {
      return contiguous_ ? BatchContiguity::Contiguous : BatchContiguity::Noncontiguous;
    }

   private:
    bool contiguous_ = false;
    bool forced_ = false;
  };


  // Forward declarations in signature, beware
  friend void MakeSampleView(class SampleWorkspace &sample, class Workspace &batch,
                             int data_idx, int thread_idx);
  friend void FixBatchPropertiesConsistency(class Workspace &ws, bool contiguous);
  friend class test::TensorListVariableBatchSizeTest_UpdatePropertiesFromSamples_Test;

  auto &tensor_handle(size_t pos) {
    return tensors_[pos];
  }

  auto &tensor_handle(size_t pos) const {
    return tensors_[pos];
  }

  template <typename T>
  void SetupLikeImpl(const T &other) {
    DALI_ENFORCE(!has_data(),
                 "Batch object can be initialized this way only when it isn't allocated.");
    set_type(other.type());
    set_sample_dim(other.shape().sample_dim());
    SetLayout(other.GetLayout());
    set_pinned(other.is_pinned());
    set_order(other.order());
    set_device_id(other.device_id());
  }

  /**
   * @brief Internal change of contiguity. Unconditionally make the batch non-contiguous.
   * Assumes that the state_ will be adjusted separately
   */
  void DoMakeNoncontiguous();

  /**
   * @brief After RunImpl(SampleWorkspace&) operated on individual samples without propagating
   * the allocation metadata back to the the batch structure, take that metadata from the samples
   * and update it in TensorList.
   *
   * @param contiguous if the Tensor was previously preallocated and should remain contiguous
   * or be treated as non-contiguous set of individual samples.
   */
  void UpdatePropertiesFromSamples(bool contiguous);

  /**
   * @brief Adjust the tensors_ member size, so they can be used with setters for individual
   * tensors. Bring them back to scope while preserving the allocation (if possible) in a manner
   * that is compatible with the current batch state (like type and dimensionality).
   */
  void resize_tensors(int size);

  /**
   * @brief When bringing the tensors_ back to scope, fill it with correct allocation metadata
   */
  void setup_tensor_allocation(int index);

  /**
   * @brief When using one contiguous allocation, rebuild the view Tensors (sharing part of the
   * contiguous buffer) that represent samples.
   */
  void recreate_views();

  /**
   * @brief Check if the metadata provided for new sample match the ones currently set for the batch
   *
   * When setting new sample, the source shape doesn't matter as it is adjusted for individual
   * sample.
   *
   * When setting new sample the `shape_` must be adjusted.
   *
   * @param error_suffix Additional description added to the error message
   */
  void VerifySampleShareCompatibility(DALIDataType type, int sample_dim, TensorLayout layout,
                                      bool pinned, int device_id,
                                      const std::string &error_suffix = ".");

  /**
   * @brief Check if the metadata provided for new sample match the ones currently set for the batch
   *
   * When copying new sample, pinned status and order of source and destination buffer can be
   * different. Necessary synchronization is handled by the copy itself.
   *
   * When copying new sample the `shape_` must be adjusted.
   *
   * @param error_suffix Additional description added to the error message
   */
  void VerifySampleCopyCompatibility(DALIDataType type, int sample_dim, TensorLayout layout,
                                     const TensorShape<> &current_shape,
                                     const TensorShape<> &new_shape,
                                     const std::string &error_suffix = ".");

  // Memory backing
  Buffer<Backend> contiguous_buffer_;
  std::weak_ptr<void> buffer_bkp_;
  // Memory, sample aliases and metadata
  // TODO(klecki): Remove SampleWorkspace (only place where we actually need those Tensor objects)
  // and swap to plain Buffer instead of using actual Tensors.
  std::vector<Tensor<Backend>> tensors_;

  // State and metadata that should be uniform regardless of the contiguity state.
  // Sample aliases should match the information stored below.
  State state_;
  int curr_num_tensors_;
  TypeInfo type_{};
  int sample_dim_ = -1;
  TensorListShape<> shape_;
  TensorLayout layout_;

  bool pinned_ = true;
  int device_ = CPU_ONLY_DEVICE_ID;
  AccessOrder order_ = AccessOrder::host();

  // So we can access the members of other TensorLists
  // with different template types
  template <typename InBackend>
  friend class TensorList;

  /** @defgroup AccessorFunctions Fallback for accessing pointers owning the samples
   * Fallback access to contiguous data or samples of the batch. It should not be used for regular
   * processing, intended mostly for batches that were made sure to be contiguous (mainly
   * for pipeline outputs).
   * @{
   */
  /**
   * @brief Return an un-typed pointer to the underlying storage.
   * The TensorList must be either empty or have a valid type and be contiguous.
   */
  friend void *unsafe_raw_mutable_data(TensorList<Backend> &batch) {
    DALI_ENFORCE(batch.IsContiguous(), "Data pointer can be obtain only for contiguous batch.");
    return batch.contiguous_buffer_.raw_mutable_data();
  }

  /**
   * @brief Return an un-typed const pointer to the underlying storage.
   * The TensorList must be either empty or have a valid type and be contiguous.
   */
  friend const void *unsafe_raw_data(const TensorList<Backend> &batch) {
    DALI_ENFORCE(batch.IsContiguous(), "Data pointer can be obtain only for contiguous batch.");
    return batch.contiguous_buffer_.raw_data();
  }

  /**
   * @brief Return the shared pointer, that we can use to correctly share the ownership of sample
   * with.
   * Sample 0 is aliased with the whole buffer, if it is contiguous.
   */
  friend shared_ptr<void> unsafe_sample_owner(TensorList<Backend> &batch, int sample_idx) {
    // create new aliasing pointer to current data allocation, so we share the use count
    // and the deleter correctly.
    if (batch.IsContiguous()) {
      return {batch.contiguous_buffer_.get_data_ptr(), batch.raw_mutable_tensor(sample_idx)};
    } else {
      return batch.tensors_[sample_idx].get_data_ptr();
    }
  }

  /**
   * @brief Return the shared pointer, that we can use to correctly share the ownership of batch
   * with.
   * Only allowed for contiguous batch, in typical scenario it is equivalent to
   * unsafe_sample_owner(batch, 0)
   */
  friend shared_ptr<void> unsafe_owner(TensorList<Backend> &batch) {
    DALI_ENFORCE(batch.IsContiguous(),
                 "Data owner pointer can be obtain only for contiguous TensorList.");
    return batch.contiguous_buffer_.get_data_ptr();
  }

  /** @} */  // end of AccessorFunctions
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_LIST_H_
