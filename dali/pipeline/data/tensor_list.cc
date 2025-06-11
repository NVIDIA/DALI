// Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>

#include "dali/core/access_order.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/tensor_layout.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/types.h"

namespace dali {
namespace copy_impl {

/**
 * @defgroup copy_impl Helper code for copying batches
 * The functions used as scaffolding for the synchronization of order for source and destination
 * buffers and extract the pointers from contiguous and non-contiguous batches.
 *
 * The usage is expected to be as follows:
 * 1. SyncBefore
 * 2. Resize the destination buffer(s)
 * 3. SyncAfterResize
 * 4. Use the CopyImpl - it can copy between batch and a single contiguous allocation, assuming both
 *    batches are correctly resized already
 * 5. SyncAfter
 *
 * @{
 */
/**
 * @brief Pick the order for Copy to be run on and synchronize
 *
 * The copy ordering can be:
 * - explict, as specified in `order`
 * - the one from `src_order`, if set
 * - the one from `dst_order`
 * @return copy_order - order on which we will do the copy
 */
AccessOrder SyncBefore(AccessOrder dst_order, AccessOrder src_order, AccessOrder order) {
  if (!order)
    order = src_order ? src_order : dst_order;

  // The destination buffer must be ready to be overwritten
  order.wait(dst_order);
  // The source buffer must be ready to cosume
  order.wait(src_order);

  return order;
}


/**
 * @brief Wait for the copy to finish in the order of the dst buffer.
 */
void SyncAfter(AccessOrder dst_order, AccessOrder copy_order) {
  dst_order.wait(copy_order);
}

/**
 * @brief Copy between two non-contiguous batches
 * Assumes matching shapes and types
 */
template <typename DstBackend, typename SrcBackend, template <typename> typename DstBatch,
          template <typename> typename SrcBatch>
void CopySamplewiseImpl(DstBatch<DstBackend> &dst, const SrcBatch<SrcBackend> &src,
                        const TypeInfo &type_info, AccessOrder order,
                        bool use_copy_kernel = false) {
  auto num_samples = src.num_samples();
  BatchVector<const void *> srcs;
  srcs.reserve(num_samples);
  BatchVector<void *> dsts;
  dsts.reserve(num_samples);
  BatchVector<Index> sizes;
  sizes.reserve(num_samples);
  for (int i = 0; i < num_samples; i++) {
    dsts.push_back(dst.raw_mutable_tensor(i));
    srcs.push_back(src.raw_tensor(i));
    sizes.push_back(src.shape()[i].num_elements());
  }

  type_info.Copy<SrcBackend, DstBackend>(dsts.data(), srcs.data(), sizes.data(), num_samples,
                                         order.stream(), use_copy_kernel);
}


/**
 * @brief Copy to non-contiguous batch from contiguous source.
 * Assumes matching shapes and type.
 */
template <typename DstBackend, typename SrcBackend, template <typename> typename DstBatch>
void CopySamplewiseImpl(DstBatch<DstBackend> &dst, const void *src, const TypeInfo &type_info,
                        AccessOrder order, bool use_copy_kernel = false) {
  auto num_samples = dst.num_samples();
  BatchVector<void *> dsts;
  dsts.reserve(num_samples);
  BatchVector<Index> sizes;
  sizes.reserve(num_samples);
  for (int i = 0; i < num_samples; i++) {
    dsts.push_back(dst.raw_mutable_tensor(i));
    sizes.push_back(dst.shape()[i].num_elements());
  }

  type_info.Copy<DstBackend, SrcBackend>(dsts.data(), src, sizes.data(), num_samples,
                                         order.stream(), use_copy_kernel);
}


/**
 * @brief Copy from non-contiguous batch to contiguous destination.
 * Assumes matching shapes and types.
 */
template <typename DstBackend, typename SrcBackend, template <typename> typename SrcBatch>
void CopySamplewiseImpl(void *dst, const SrcBatch<SrcBackend> &src, const TypeInfo &type_info,
                        AccessOrder order, bool use_copy_kernel = false) {
  auto num_samples = src.num_samples();
  BatchVector<const void *> srcs;
  srcs.reserve(num_samples);
  BatchVector<Index> sizes;
  sizes.reserve(num_samples);
  for (int i = 0; i < num_samples; i++) {
    srcs.push_back(src.raw_tensor(i));
    sizes.push_back(src.shape()[i].num_elements());
  }

  type_info.Copy<DstBackend, SrcBackend>(dst, srcs.data(), sizes.data(), num_samples,
                                         order.stream(), use_copy_kernel);
}

/**
 * @brief Copy from batch to batch, detecting the contiguous/noncontiguous setups.
 * Assumes matching shapes and types.
 */
template <typename DstBackend, typename SrcBackend, template <typename> typename DstBatch,
          template <typename> typename SrcBatch>
void CopyImpl(DstBatch<DstBackend> &dst, const SrcBatch<SrcBackend> &src, const TypeInfo &type_info,
              AccessOrder copy_order, bool use_copy_kernel = false) {
  if (dst.IsContiguous() && src.IsContiguous()) {
    type_info.Copy<DstBackend, SrcBackend>(contiguous_raw_mutable_data(dst),
                                           contiguous_raw_data(src),
                                           dst.shape().num_elements(), copy_order.stream(),
                                           use_copy_kernel);
  } else if (dst.IsContiguous() && !src.IsContiguous()) {
    copy_impl::CopySamplewiseImpl<DstBackend, SrcBackend>(contiguous_raw_mutable_data(dst), src,
                                                          type_info, copy_order, use_copy_kernel);
  } else if (!dst.IsContiguous() && src.IsContiguous()) {
    copy_impl::CopySamplewiseImpl<DstBackend, SrcBackend>(dst, contiguous_raw_data(src), type_info,
                                                          copy_order, use_copy_kernel);
  } else {
    copy_impl::CopySamplewiseImpl<DstBackend, SrcBackend>(dst, src, type_info, copy_order,
                                                          use_copy_kernel);
  }
}

/** @} */  // end of copy_impl

}  // namespace copy_impl

template <typename Backend>
TensorList<Backend>::TensorList() : curr_num_tensors_(0) {}


template <typename Backend>
TensorList<Backend>::TensorList(int batch_size) : curr_num_tensors_(0) {
  // We don't use negative batch size through DALI, and by default we wanted batch to
  // not do any initial allocation unless actual shape is provided.
  // As the -1 and 0 sample dims (the latter being reserved for scalar case already),
  // are not widely supported within DALI codebase, we use `dim=1` and end up with samples
  // of shape {0}.
  set_sample_dim(1);
  resize_tensors(batch_size);
}


template <typename Backend>
TensorList<Backend>::TensorList(TensorList<Backend> &&other) noexcept {
  *this = std::move(other);
}

template <typename Backend>
TensorList<Backend>::~TensorList() {
  Reset();
}


template <typename Backend>
TensorList<Backend> &TensorList<Backend>::operator=(TensorList<Backend> &&other) noexcept {
  if (&other != this) {
    contiguous_buffer_ = std::move(other.contiguous_buffer_);
    tensors_ = std::move(other.tensors_);

    state_ = other.state_;
    curr_num_tensors_ = other.curr_num_tensors_;
    type_ = other.type_;
    sample_dim_ = other.sample_dim_;
    shape_ = std::move(other.shape_);
    layout_ = std::move(other.layout_);
    pinned_ = other.pinned_;
    order_ = other.order_;
    device_ = other.device_;

    other.Reset();
  }
  return *this;
}


template <typename Backend>
void TensorList<Backend>::VerifySampleShareCompatibility(DALIDataType type, int sample_dim,
                                                         TensorLayout layout, bool pinned,
                                                         int device_id,
                                                         const std::string &error_suffix) {
  // Checks in the order of class members
  DALI_ENFORCE(this->type() == type,
               make_string("Sample must have the same type as the target batch, current: ",
                           this->type(), ", new: ", type, error_suffix));

  DALI_ENFORCE(this->sample_dim() == sample_dim,
               make_string("Sample must have the same sample dim as the target batch, current: ",
                           this->sample_dim(), ", new: ", sample_dim, error_suffix));

  DALI_ENFORCE(this->GetLayout() == layout || layout.empty(),
               make_string("Sample must have the same layout as the target batch current: ",
                           this->GetLayout(), ", new: ", layout, " or come with empty layout ",
                           error_suffix));

  DALI_ENFORCE(this->is_pinned() == pinned,
               make_string("Sample must have the same pinned status as target batch, current: ",
                           this->is_pinned(), ", new: ", pinned, error_suffix));

  DALI_ENFORCE(this->device_id() == device_id,
               make_string("Sample must have the same device id as target batch, current: ",
                           this->device_id(), ", new: ", device_id, error_suffix));
}


template <typename Backend>
void TensorList<Backend>::SetSample(int sample_idx, const TensorList<Backend> &src,
                                    int src_sample_idx) {
  // Bounds check
  assert(sample_idx >= 0 && sample_idx < curr_num_tensors_);
  assert(src_sample_idx >= 0 && src_sample_idx < src.curr_num_tensors_);
  // Setting any individual sample converts the batch to non-contiguous mode
  MakeNoncontiguous();
  if (&src.tensors_[src_sample_idx] == &tensors_[sample_idx])
    return;
  VerifySampleShareCompatibility(src.type(), src.shape().sample_dim(), src.GetLayout(),
                                 src.is_pinned(), src.device_id(),
                                 make_string(" for source sample idx: ", src_sample_idx,
                                             " and target sample idx: ", sample_idx, "."));

  shape_.set_tensor_shape(sample_idx, src.shape().tensor_shape_span(src_sample_idx));

  // Setting a new share overwrites the previous one - so we can safely assume that even if
  // we had a sample sharing into TL, it will be overwritten
  tensors_[sample_idx].ShareData(src.tensors_[src_sample_idx]);
  // As the order was simply copied over, we have to fix it back.
  // We will be accessing it in order of this buffer, so we need to wait for all the work
  // from the "incoming" src order.
  tensors_[sample_idx].set_order(order(), false);
  order().wait(src.order());

  if (src.GetLayout().empty() && !GetLayout().empty()) {
    tensors_[sample_idx].SetLayout(GetLayout());
  }
}


template <typename Backend>
void TensorList<Backend>::SetSample(int sample_idx, const Tensor<Backend> &owner) {
  // Bounds check
  assert(sample_idx >= 0 && sample_idx < curr_num_tensors_);
  // Setting any individual sample converts the batch to non-contiguous mode
  MakeNoncontiguous();
  VerifySampleShareCompatibility(owner.type(), owner.shape().sample_dim(), owner.GetLayout(),
                                 owner.is_pinned(), owner.device_id(),
                                 make_string(" for sample idx: ", sample_idx, "."));

  shape_.set_tensor_shape(sample_idx, owner.shape());

  // Setting a new share overwrites the previous one - so we can safely assume that even if
  // we had a sample sharing into TL, it will be overwritten
  tensors_[sample_idx].ShareData(owner);
  // As the order was simply copied over, we have to fix it back.
  // We will be accessing it in order of this buffer, so we need to wait for all the work
  // from the "incoming" src order.
  tensors_[sample_idx].set_order(order(), false);
  order().wait(owner.order());

  if (owner.GetLayout().empty() && !GetLayout().empty()) {
    tensors_[sample_idx].SetLayout(GetLayout());
  }
}


template <typename Backend>
void TensorList<Backend>::SetSample(int sample_idx, shared_ptr<void> ptr, size_t bytes,
                                    bool pinned, const TensorShape<> &shape, DALIDataType type,
                                    int device_id, AccessOrder order, const TensorLayout &layout) {
  // Bounds check
  assert(sample_idx >= 0 && sample_idx < curr_num_tensors_);
  // Setting any individual sample converts the batch to non-contiguous mode
  MakeNoncontiguous();
  VerifySampleShareCompatibility(type, shape.sample_dim(), layout, pinned, device_id,
                                 make_string(" for sample idx: ", sample_idx, "."));

  DALI_ENFORCE(!IsContiguous());
  shape_.set_tensor_shape(sample_idx, shape);

  // Setting a new share overwrites the previous one - so we can safely assume that even if
  // we had a sample sharing into TL, it will be overwritten
  tensors_[sample_idx].ShareData(std::move(ptr), bytes, pinned, shape, type, device_id, order);
  // As the order was simply copied over, we have to fix it back.
  // We will be accessing it in order of this buffer, so we need to wait for all the work
  // from the "incoming" src order.
  tensors_[sample_idx].set_order(this->order(), false);
  this->order().wait(order);

  if (layout.empty() && !GetLayout().empty()) {
    tensors_[sample_idx].SetLayout(GetLayout());
  }
}


template <typename Backend>
void TensorList<Backend>::VerifySampleCopyCompatibility(DALIDataType type, int sample_dim,
                                                        TensorLayout layout,
                                                        const TensorShape<> &current_shape,
                                                        const TensorShape<> &new_shape,
                                                        const std::string &error_suffix) {
  // Checks in the order of class members
  DALI_ENFORCE(this->type() == type,
               make_string("Sample must have the same type as the target batch, current: ",
                           this->type(), ", new: ", type, error_suffix));

  DALI_ENFORCE(this->sample_dim() == sample_dim,
               make_string("Sample must have the same sample dim as the target batch, current: ",
                           this->sample_dim(), ", new: ", sample_dim, error_suffix));

  if (IsContiguous()) {
    DALI_ENFORCE(
        current_shape.num_elements() == new_shape.num_elements(),
        make_string("Sample volume must match when copying to a contiguous batch, current: ",
                    current_shape.num_elements(), ", new: ", new_shape.num_elements(),
                    error_suffix));
  }

  DALI_ENFORCE(this->GetLayout() == layout || layout.empty(),
               make_string("Sample must have the same layout as the target batch current: ",
                           this->GetLayout(), ", new: ", layout, " or come with empty layout ",
                           error_suffix));
}


template <typename Backend>
void TensorList<Backend>::CopySample(int sample_idx, const TensorList<Backend> &src,
                                     int src_sample_idx, AccessOrder order) {
  // Bounds check
  assert(sample_idx >= 0 && sample_idx < curr_num_tensors_);
  assert(src_sample_idx >= 0 && src_sample_idx < src.curr_num_tensors_);
  VerifySampleCopyCompatibility(src.type(), src.shape().sample_dim(), src.GetLayout(),
                                shape()[sample_idx], src.shape()[src_sample_idx],
                                make_string(" for source sample idx: ", src_sample_idx,
                                            " and target sample idx: ", sample_idx, "."));

  shape_.set_tensor_shape(sample_idx, src.shape()[src_sample_idx]);
  tensors_[sample_idx].Copy(src.tensors_[src_sample_idx], order);
  if (src.GetLayout().empty() && !GetLayout().empty()) {
    tensors_[sample_idx].SetLayout(GetLayout());
  }
}


template <typename Backend>
void TensorList<Backend>::CopySample(int sample_idx, const Tensor<Backend> &src,
                                     AccessOrder order) {
  // Bounds check
  assert(sample_idx >= 0 && sample_idx < curr_num_tensors_);
  VerifySampleCopyCompatibility(src.type(), src.shape().sample_dim(), src.GetLayout(),
                                shape()[sample_idx], src.shape(),
                                make_string(" for sample idx: ", sample_idx, "."));

  shape_.set_tensor_shape(sample_idx, src.shape());
  tensors_[sample_idx].Copy(src, order);
  if (src.GetLayout().empty() && !GetLayout().empty()) {
    tensors_[sample_idx].SetLayout(GetLayout());
  }
}


template <typename Backend>
void TensorList<Backend>::set_sample_dim(int sample_dim) {
  DALI_ENFORCE(
      !has_data(),
      "Setting sample dim is not allowed when batch is already allocated, use Resize instead.");
  sample_dim_ = sample_dim;
  shape_.resize(shape_.num_samples(), sample_dim);
}


template <typename Backend>
size_t TensorList<Backend>::nbytes() const noexcept {
  if (IsContiguous()) {
    return contiguous_buffer_.nbytes();
  }
  // else
  size_t nbytes = 0;
  for (const auto &t : tensors_) {
    nbytes += t.nbytes();
  }
  return nbytes;
}


template <typename Backend>
size_t TensorList<Backend>::capacity() const noexcept {
  if (IsContiguous()) {
    return contiguous_buffer_.capacity();
  }
  // else
  size_t capacity = 0;
  for (const auto &t : tensors_) {
    capacity += t.capacity();
  }
  return capacity;
}


template <typename Backend>
std::vector<size_t> TensorList<Backend>::_chunks_nbytes() const {
  if (IsContiguous()) {
    return {contiguous_buffer_.nbytes()};
  }
  // else
  std::vector<size_t> result(tensors_.size());
  for (size_t i = 0; i < tensors_.size(); i++) {
    result[i] = tensors_[i].nbytes();
  }
  return result;
}


template <typename Backend>
std::vector<size_t> TensorList<Backend>::_chunks_capacity() const {
  if (IsContiguous()) {
    return {contiguous_buffer_.capacity()};
  }
  // else
  std::vector<size_t> result(tensors_.size());
  for (size_t i = 0; i < tensors_.size(); i++) {
    result[i] = tensors_[i].capacity();
  }
  return result;
}

template <typename Backend>
void TensorList<Backend>::set_order(AccessOrder order, bool synchronize) {
  DALI_ENFORCE(order, "Resetting order to an empty one is not supported");

  if (this->order() == order)
    return;

  // Optimization: synchronize only once, if needed.
  if (this->order().is_device() && order && synchronize) {
    bool need_sync = contiguous_buffer_.has_data();
    if (!need_sync) {
      for (const auto &t : tensors_) {
        if (t.has_data()) {
          need_sync = true;
          break;
        }
      }
    }
    if (need_sync)
      order.wait(this->order());
  }
  contiguous_buffer_.set_order(order, false);
  for (auto &t : tensors_)
    t.set_order(order, false);
  order_ = order;
}


template <typename Backend>
SampleView<Backend> TensorList<Backend>::operator[](size_t pos) {
  DALI_ENFORCE(pos < static_cast<size_t>(curr_num_tensors_), "Out of bounds access");
  return {tensors_[pos].raw_mutable_data(), shape().tensor_shape_span(pos), tensors_[pos].type()};
}


template <typename Backend>
ConstSampleView<Backend> TensorList<Backend>::operator[](size_t pos) const {
  DALI_ENFORCE(pos < static_cast<size_t>(curr_num_tensors_), "Out of bounds access");
  return {tensors_[pos].raw_data(), shape().tensor_shape_span(pos), tensors_[pos].type()};
}


template <typename Backend>
void TensorList<Backend>::Resize(const TensorListShape<> &new_shape, DALIDataType new_type,
                                 BatchContiguity state) {
  DALI_ENFORCE(IsValidType(new_type),
               "TensorList cannot be resized with invalid type. To zero out the TensorList "
               "Reset() can be used.");
  if (state_.Update(state)) {
    if (!state_.IsContiguous()) {
      // As we updated the state to noncontiguous, we need to detach the buffers
      DoMakeNoncontiguous();
    }
  }

  // Resize the tensors_ and setup the allocation metadata on the tensors, just in case
  // we will be resizing them. Rest of their metadata (like shape and type) will be updated
  // by either the recreate_views or Tensor::Resize.
  int old_size = tensors_.size();
  if (old_size < new_shape.num_samples()) {
    tensors_.resize(new_shape.num_samples());
  }

  for (int i = old_size; i < new_shape.num_samples(); i++) {
    setup_tensor_allocation(i);
  }
  curr_num_tensors_ = new_shape.num_samples();

  if (type_.id() != new_type) {
    type_ = TypeTable::GetTypeInfo(new_type);
  }

  sample_dim_ = new_shape.sample_dim();
  shape_ = new_shape;

  bool should_coalesce = [&]() {
    if (!state_.IsContiguous() && state_.IsForced()) {
      return false;  // someone needs non-contiguous data
    }
    if (state_.IsContiguous()) {
      return false;  // already coalesced
    }
    if (state == BatchContiguity::Noncontiguous) {
      return false;  // we were requested to keep it non-contiguous
    }
    // TODO(klecki): as an alternative, coalesce every time?
    for (int i = 0; i < curr_num_tensors_; i++) {
      if (tensors_[i].capacity() < volume(new_shape.tensor_shape_span(i)) * type_.size()) {
        return true;  // we will be reallocating either way, let's coalesce
      }
    }
    return false;
  }();

  if (should_coalesce) {
    state_.Update(BatchContiguity::Contiguous);
  }

  if (state_.IsContiguous()) {
    contiguous_buffer_.resize(new_shape.num_elements(), new_type);
    order_ = contiguous_buffer_.order();  // propagate order after allocation, it might have changed
    device_ = contiguous_buffer_.device_;
    recreate_views();
    return;
  }

  for (int i = 0; i < curr_num_tensors_; i++) {
    tensors_[i].Resize(new_shape[i], new_type);
  }

  if (curr_num_tensors_ > 0) {
    order_ = tensors_[0].order();
    device_ = tensors_[0].device_id();
  }
  SetLayout(GetLayout());
}


template <typename Backend>
void TensorList<Backend>::ResizeSample(int sample_idx, const TensorShape<> &new_shape) {
  DALI_ENFORCE(IsValidType(type()),
               "Sample in TensorList cannot be resized with invalid type. Set the type first for "
               "the whole TensorList using set_type or Resize.");
  DALI_ENFORCE(sample_dim() == new_shape.sample_dim(),
               "Sample in TensorList cannot be resized with non-compatible batch dimension. Use "
               "set_sample_dim or Resize to set correct sample dimension for the whole batch.");
  // Bounds check
  assert(sample_idx >= 0 && sample_idx < curr_num_tensors_);
  // Resizing any individual sample converts the batch to non-contiguous mode
  MakeNoncontiguous();
  shape_.set_tensor_shape(sample_idx, new_shape);
  tensors_[sample_idx].Resize(new_shape);
}


template <typename Backend>
void TensorList<Backend>::SetSize(int new_size) {
  DALI_ENFORCE(new_size >= 0, make_string("Incorrect size: ", new_size));
  resize_tensors(new_size);
}


template <typename Backend>
void TensorList<Backend>::set_type(DALIDataType new_type_id) {
  DALI_ENFORCE(new_type_id != DALI_NO_TYPE, "new_type must be valid type.");
  if (type_.id() == new_type_id)
    return;
  DALI_ENFORCE(type_.id() == new_type_id || (!has_data() || type_.id() == DALI_NO_TYPE),
               make_string("set_type cannot be used to change the current type - it is not "
                           "allowed to cause allocations. Currently set type: '",
                           type_.id(), "' trying to set: '", new_type_id,
                           "'. You may change the current type using Resize or by"
                           " calling Reset first."));
  contiguous_buffer_.set_type(new_type_id);
  type_ = TypeTable::GetTypeInfo(new_type_id);
  if (state_.IsContiguous()) {
    recreate_views();
  } else {
    for (auto &t : tensors_) {
      t.set_type(new_type_id);
    }
  }
}

template <typename Backend>
void TensorList<Backend>::SetLayout(const TensorLayout &layout) {
  for (auto &t : tensors_) {
    t.SetLayout(layout);
  }
  layout_ = layout;
}


template <typename Backend>
void TensorList<Backend>::SetSkipSample(int idx, bool skip_sample) {
  tensors_[idx].SetSkipSample(skip_sample);
}


template <typename Backend>
void TensorList<Backend>::SetSourceInfo(int idx, const std::string &source_info) {
  tensors_[idx].SetSourceInfo(source_info);
}

template <typename Backend>
const DALIMeta &TensorList<Backend>::GetMeta(int idx) const {
  assert(idx < curr_num_tensors_);
  return tensors_[idx].GetMeta();
}


template <typename Backend>
void TensorList<Backend>::SetMeta(int idx, const DALIMeta &meta) {
  assert(idx < curr_num_tensors_);
  DALI_ENFORCE(GetLayout() == meta.GetLayout(),
               make_string("Sample must have the same layout as the target batch, current: ",
                           GetLayout(), " new: ", meta.GetLayout(), " for sample ", idx, "."));
  tensors_[idx].SetMeta(meta);
}


template <typename Backend>
void TensorList<Backend>::set_pinned(bool pinned) {
  contiguous_buffer_.set_pinned(pinned);
  for (auto &t : tensors_) {
    t.set_pinned(pinned);
  }
  pinned_ = pinned;
}

template <typename Backend>
void TensorList<Backend>::set_device_id(int device_id) {
  contiguous_buffer_.set_device_id(device_id);
  for (auto &t : tensors_) {
    t.set_device_id(device_id);
  }
  device_ = device_id;
}

template <typename Backend>
void TensorList<Backend>::reserve(size_t total_bytes) {
  int batch_size_bkp = curr_num_tensors_;
  if (!state_.IsContiguous()) {
    tensors_.clear();
    resize_tensors(0);
  }
  state_.Setup(BatchContiguity::Contiguous);
  contiguous_buffer_.reserve(total_bytes);
  if (IsValidType(type_)) {
    resize_tensors(batch_size_bkp);
    recreate_views();
  }
}


template <typename Backend>
void TensorList<Backend>::reserve(size_t bytes_per_sample, int batch_size) {
  assert(batch_size > 0);
  state_.Setup(BatchContiguity::Noncontiguous);
  resize_tensors(batch_size);
  for (int i = 0; i < curr_num_tensors_; i++) {
    tensors_[i].reserve(bytes_per_sample);
  }
}

template <typename Backend>
void TensorList<Backend>::recreate_views() {
  // precondition: type, shape are configured
  uint8_t *sample_ptr = static_cast<uint8_t *>(contiguous_buffer_.raw_mutable_data());
  int64_t num_samples = shape().num_samples();
  auto &data_ptr = contiguous_buffer_.get_data_ptr();
  for (int64_t i = 0; i < num_samples; i++) {
    // or any other way
    auto tensor_size = shape().tensor_size(i);

    tensors_[i].ShareData(std::shared_ptr<void>(data_ptr, sample_ptr),
                          tensor_size * type_info().size(), is_pinned(), shape()[i],
                          type(), device_id(), order());
    tensors_[i].SetLayout(GetLayout());
    sample_ptr += tensor_size * type_info().size();
  }
}


template <typename Backend>
void TensorList<Backend>::SetContiguity(BatchContiguity state) {
  if (state == BatchContiguity::Automatic) {
    // remove the force, keep the current state information
    state_.Setup(state_.Get(), false);
    return;
  }
  DALI_ENFORCE(state_.Get() == state || !has_data(),
               "Contiguous or non-contiguous mode cannot be set to already allocated buffer.");
  state_.Setup(state, true);
}


template <typename Backend>
void TensorList<Backend>::MakeContiguous(std::weak_ptr<void> owner) {
  if (state_.IsContiguous()) {
    return;
  }
  DALI_FAIL("Coalescing the buffer to Contiguous state is not yet implemented.");
}


template <typename Backend>
void TensorList<Backend>::MakeNoncontiguous() {
  if (!state_.IsContiguous()) {
    return;
  }

  state_.Update(BatchContiguity::Noncontiguous);
  DoMakeNoncontiguous();
}


template <typename Backend>
void TensorList<Backend>::DoMakeNoncontiguous() {
  auto &contiguous_ptr = contiguous_buffer_.get_data_ptr();
  for (auto &t : tensors_) {
    // If the Tensor was aliasing the contiguous buffer, mark it as not sharing any data.
    // This will allow for the individual buffers to be resized.
    // The downside of this is we may keep the big contiguous buffer until all individual
    // samples are replaced.
    if (same_managed_object(contiguous_ptr, t.data_)) {
      t.detach();
    }
  }
  contiguous_buffer_.reset();
}


template <typename Backend>
void TensorList<Backend>::Reset() {
  if (contiguous_buffer_.data_) {
    // Optimization: prevent per-sample synchronization for samples sharing the buffer
    // with the batch.
    for (auto &t : tensors_) {
      if (t.order_ == order_ && same_managed_object(t.data_, contiguous_buffer_.data_)) {
        // reset the internal pointer - we're still holding a reference
        t.data_.reset();
      }
    }

    contiguous_buffer_.reset();
  }

  // Clear the tensors - after the code above they might be in an inconsistent state.
  tensors_.clear();

  curr_num_tensors_ = 0;
  type_ = {};
  sample_dim_ = -1;
  shape_ = {};
  layout_ = "";
  // N.B. state_, pinned_, order_, device_ and ready_ are not reset here, as they might be
  // previously set up via the executor - TODO(klecki) - consider if we want to keep this behaviour
}


template <typename Backend>
template <typename SrcBackend>
void TensorList<Backend>::Copy(const TensorList<SrcBackend> &src, AccessOrder order,
                               bool use_copy_kernel) {
  if (!IsValidType(src.type())) {
    assert(!src.has_data() && "It is not possible to have data without valid type.");
    Reset();
    SetLayout(src.GetLayout());
    // no copying to do
    return;
  }
  if (std::is_same_v<Backend, CPUBackend> &&
      std::is_same_v<SrcBackend, CPUBackend>) {
    DALI_ENFORCE(!order.is_device(),
      "Cannot run a host-to-host copy on a device stream.");
    if (!order)
      order = AccessOrder::host();
  }

  Resize(src.shape(), src.type());
  // After resize the state_, curr_num_tensors_, type_, sample_dim_, shape_ (and pinned)
  // postconditions are met, as well as the buffers are correctly adjusted.

  auto copy_order = copy_impl::SyncBefore(this->order(), src.order(), order);

  use_copy_kernel &= (std::is_same<SrcBackend, GPUBackend>::value || src.is_pinned()) &&
                     (std::is_same<Backend, GPUBackend>::value || this->is_pinned());

  copy_impl::CopyImpl(*this, src, this->type_info(), copy_order, use_copy_kernel);

  // Update the layout and other metadata
  SetLayout(src.GetLayout());
  for (int i = 0; i < curr_num_tensors_; i++) {
    SetMeta(i, src.GetMeta(i));
  }

  copy_impl::SyncAfter(this->order(), copy_order);
}


template <typename Backend>
void TensorList<Backend>::ShareData(const TensorList<Backend> &tl) {
  if (this == &tl)
    return;

  // We need not just the pointer values, but also the underlying managed objects to be the same
  // to consider this an identity operation.

  bool same_data = same_shared_ptr(contiguous_buffer_.data_, tl.contiguous_buffer_.data_);
  if (!tl.IsContiguous() && same_data) {
    if (num_samples() == tl.num_samples()) {
      for (int i = 0; i < num_samples(); i++) {
        if (!same_shared_ptr(tensors_[i].data_, tl.tensors_[i].data_)) {
          same_data = false;
          break;
        }
      }
    } else {
      same_data = false;
    }
  }

  // if the data is the same, there's no point in resetting the buffer (and possibly synchronizing)
  if (!same_data)
    Reset();

  state_ = tl.state_;
  curr_num_tensors_ = tl.curr_num_tensors_;
  type_ = tl.type_;
  sample_dim_ = tl.sample_dim_;
  shape_ = tl.shape_;
  layout_ = tl.layout_;
  pinned_ = tl.pinned_;
  order_ = tl.order_;
  device_ = tl.device_;
  ready_ = tl.ready_;

  if (tl.IsContiguous()) {
    if (!same_data)
      contiguous_buffer_.ShareData(tl.contiguous_buffer_);
    tensors_.resize(shape().num_samples());
    recreate_views();
  } else {
    if (!same_data) {
      int batch_size = tl.num_samples();
      tensors_.resize(shape().num_samples());
      for (int i = 0; i < batch_size; i++) {
        tensors_[i].ShareData(tl.tensors_[i]);
      }
    }
  }

  SetLayout(tl.GetLayout());
  for (int i = 0; i < curr_num_tensors_; i++) {
    SetMeta(i, tl.GetMeta(i));
  }
}


// This is to check if we are actually laid down in contiguous memory
template <typename Backend>
bool TensorList<Backend>::IsContiguousInMemory() const {
  if (num_samples() == 0 || shape().num_elements() == 0) {
    return true;
  }
  // If we are using contiguous representation in this case we can safely return
  if (IsContiguous()) {
    return true;
  }
  const uint8_t *base_ptr = static_cast<const uint8_t *>(tensors_[0].raw_data());
  size_t size = type_info().size();

  for (int i = 0; i < num_samples(); ++i) {
    if (base_ptr != tensors_[i].raw_data()) {
      return false;
    }
    base_ptr += shape_[i].num_elements() * size;
  }
  return true;
}


template <typename Backend>
bool TensorList<Backend>::IsDenseTensor() const {
  return IsContiguousInMemory() && is_uniform(shape());
}


template <typename Backend>
Tensor<Backend> TensorList<Backend>::AsReshapedTensor(const TensorShape<> &new_shape) {
  DALI_ENFORCE(new_shape.num_elements() == 0 || num_samples() > 0,
               "To create a non-empty view Tensor, the batch must not be empty.");
  DALI_ENFORCE(IsValidType(type()),
               "To create a view Tensor, the batch must have a valid data type.");
  DALI_ENFORCE(
      shape().num_elements() == new_shape.num_elements(),
      make_string("To create a view Tensor, requested shape need to have the same volume as the "
                  "batch, requested: ",
                  new_shape.num_elements(), " expected: ", shape().num_elements()));
  DALI_ENFORCE(IsContiguousInMemory(),
               "To create a view Tensor, the batch must be in contiguous memory.");

  Tensor<Backend> result;

  shared_ptr<void> ptr;
  if (num_samples() > 0) {
    ptr = unsafe_sample_owner(*this, 0);
  } else if (IsContiguous()) {
    ptr = unsafe_owner(*this);
  } else {
    ptr = nullptr;
  }

  result.ShareData(std::move(ptr), capacity(), is_pinned(),
                   new_shape, type(), device_id(), order(), ready_);

  auto result_layout = GetLayout();
  if (result_layout.ndim() + 1 == new_shape.sample_dim()) {
    result_layout = TensorLayout("N") + result_layout;
    result.SetLayout(result_layout);
  }
  return result;
}


template <typename Backend>
Tensor<Backend> TensorList<Backend>::AsTensor() {
  DALI_ENFORCE(IsDenseTensor(),
               "The batch must be representable as a tensor - it must have uniform shape and be "
               "allocated in contiguous memory.");
  if (shape().num_samples() == 0) {
    DALI_ENFORCE(sample_dim() > 0,
                 "To convert empty batch to a Tensor, valid dimensionality must be set");
    return AsReshapedTensor(TensorShape<>::empty_shape(sample_dim()));
  }
  return AsReshapedTensor(shape_cat(shape().num_samples(), shape()[0]));
}


template <typename Backend>
void TensorList<Backend>::ShareData(shared_ptr<void> ptr, size_t bytes, bool pinned,
                                    const TensorListShape<> &shape, DALIDataType type,
                                    int device_id, AccessOrder order, const TensorLayout &layout,
                                    CUDASharedEvent ready) {
  contiguous_buffer_.set_backing_allocation(std::move(ptr), bytes, pinned,
                                            type, shape.num_elements(),
                                            device_id, order);
  tensors_.clear();
  tensors_.resize(shape.num_samples());

  state_.Update(BatchContiguity::Contiguous);
  curr_num_tensors_ = shape.num_samples();
  type_ = TypeTable::GetTypeInfo(type);
  sample_dim_ = shape.sample_dim();
  shape_ = shape;
  layout_ = layout;
  pinned_ = pinned;
  device_ = device_id;
  ready_ = ready;
  if (order)
    order_ = order;
  recreate_views();
}


template <typename Backend>
void TensorList<Backend>::setup_tensor_allocation(int index) {
  if (tensors_[index].has_data() && tensors_[index].is_pinned() != is_pinned()) {
    tensors_[index].Reset();
  }
  if (!tensors_[index].has_data()) {
    tensors_[index].set_pinned(is_pinned());
  }
  tensors_[index].set_device_id(device_id());
  tensors_[index].set_order(order());
}


template <typename Backend>
void TensorList<Backend>::resize_tensors(int new_size) {
  // This doesn't update with the same order as the class members are listed
  // We need to make sure everything is updated for the tensors that come back into scope
  // and we start with the pinned and order properties as they might impact future allocations.
  // next we make sure the type is consistent, and if so, introduce empty shape
  shape_.resize(new_size);
  if (new_size > curr_num_tensors_) {
    auto old_size = curr_num_tensors_;
    tensors_.resize(new_size);
    for (int i = old_size; i < new_size; i++) {
      setup_tensor_allocation(i);
      if (type() != DALI_NO_TYPE) {
        if (sample_dim_ >= 0) {
          // We can't have empty scalar.
          const auto &emptyish_shape = sample_dim() > 0 ? TensorShape<>::empty_shape(sample_dim()) :
                                                          TensorShape<>();
          tensors_[i].Resize(emptyish_shape, type());
          shape_.set_tensor_shape(i, emptyish_shape);
        } else if (type() != tensors_[i].type()) {
          tensors_[i].Reset();
          tensors_[i].set_type(type());
        }
      }
      tensors_[i].SetLayout(GetLayout());
    }
  } else if (new_size < curr_num_tensors_) {
    // TODO(klecki): Do not keep the invalidated tensors - this prevents memory hogging but
    // also gets rid of reserved memory. For now keeping the old behaviour.
    for (int i = new_size; i < curr_num_tensors_; i++) {
      if (tensors_[i].shares_data()) {
        tensors_[i].Reset();
      }
    }
  }
  curr_num_tensors_ = new_size;
}


template <typename Backend>
void TensorList<Backend>::UpdatePropertiesFromSamples(bool contiguous) {
  if (contiguous) {
    bool is_really_contiguous = true;

    const uint8_t *base_ptr = static_cast<const uint8_t *>(contiguous_buffer_.raw_data());
    size_t size = type_info().size();

    for (int i = 0; i < num_samples(); ++i) {
      if (tensors_[i].raw_data() == nullptr)
        DALI_ENFORCE(shape_[i].num_elements() == 0,
                     "Internal error: a non-empty sample has a null data pointer.");
      if (base_ptr != tensors_[i].raw_data()) {
        is_really_contiguous = false;
        break;
      }
      base_ptr += shape_[i].num_elements() * size;
    }
    DALI_ENFORCE(is_really_contiguous,
                 "Internal error: The tensor list isn't really contiguous as claimed.");
  }
  state_.Update(contiguous ? BatchContiguity::Contiguous : BatchContiguity::Noncontiguous);

  // assume that the curr_num_tensors_ is valid
  DALI_ENFORCE(curr_num_tensors_ > 0,
               "Unexpected empty output of per-sample operator. Internal DALI error.");
  for (int i = 0; i < curr_num_tensors_; i++) {
    // if tensor is empty it can be uninitialized, so find the initialized one
    if (tensors_[i].nbytes() == 0 && i != curr_num_tensors_ - 1) continue;
    type_ = tensors_[i].type_info();
    sample_dim_ = tensors_[i].shape().sample_dim();
    shape_.resize(curr_num_tensors_, sample_dim_);
    layout_ = tensors_[i].GetMeta().GetLayout();
    pinned_ = tensors_[i].is_pinned();
    order_ = tensors_[i].order();
    device_ = tensors_[i].device_id();
    contiguous_buffer_.set_order(order_);
    break;
  }
  for (int i = 0; i < curr_num_tensors_; i++) {
    if (tensors_[i].nbytes() == 0) {
      if (is_pinned() != tensors_[i].is_pinned() || order() != tensors_[i].order()) {
        tensors_[i].reset();
        tensors_[i].set_pinned(is_pinned());
        tensors_[i].set_order(order());
      }
      tensors_[i].set_type(type());
      tensors_[i].SetLayout(GetLayout());
      tensors_[i].set_device_id(device_id());
    }
    DALI_ENFORCE(type() == tensors_[i].type(),
                 make_string("Samples must have the same type, expected: ", type(),
                             " got: ", tensors_[i].type(), " at ", i, "."));
    DALI_ENFORCE(sample_dim() == tensors_[i].shape().sample_dim(),
                 make_string("Samples must have the same dimensionality, expected: ", sample_dim(),
                             " got: ", tensors_[i].shape().sample_dim(), " at ", i, "."));
    DALI_ENFORCE(GetLayout() == tensors_[i].GetLayout(),
                 make_string("Samples must have the same layout, expected: ", GetLayout(),
                             " got: ", tensors_[i].GetLayout(), " at ", i, "."));
    DALI_ENFORCE(is_pinned() == tensors_[i].is_pinned(),
                 make_string("Samples must have the same pinned status, expected: ", is_pinned(),
                             " got: ", tensors_[i].is_pinned(), " at ", i, "."));
    DALI_ENFORCE(order() == tensors_[i].order(),
                 make_string("Samples must have the same order, expected: ", order().get(), " ",
                             order().device_id(), " got: ", tensors_[i].order().get(), " ",
                             tensors_[i].order().device_id(), " at ", i, "."));
    DALI_ENFORCE(device_id() == tensors_[i].device_id(),
                 make_string("Samples must have the same device id, expected: ", device_id(),
                             " got: ", tensors_[i].device_id(), " at ", i, "."));
    shape_.set_tensor_shape(i, tensors_[i].shape());
  }
}


template <typename Backend>
bool TensorList<Backend>::has_data() const {
  if (IsContiguous()) {
    return contiguous_buffer_.has_data();
  }
  for (const auto &tensor : tensors_) {
    if (tensor.has_data()) {
      return true;
    }
  }
  return false;
}


template <typename Backend>
bool TensorList<Backend>::shares_data() const {
  // TODO(klecki): I would like to get rid of some of this
  if (IsContiguous()) {
    return contiguous_buffer_.shares_data();
  }
  for (const auto &tensor : tensors_) {
    if (tensor.shares_data() &&
        !same_managed_object(contiguous_buffer_.get_data_ptr(), tensor.get_data_ptr())) {
      return true;
    }
  }
  return false;
}

template class DLL_PUBLIC TensorList<CPUBackend>;
template class DLL_PUBLIC TensorList<GPUBackend>;
template void TensorList<CPUBackend>::Copy<CPUBackend>(const TensorList<CPUBackend> &, AccessOrder,
                                                       bool);
template void TensorList<CPUBackend>::Copy<GPUBackend>(const TensorList<GPUBackend> &, AccessOrder,
                                                       bool);
template void TensorList<GPUBackend>::Copy<CPUBackend>(const TensorList<CPUBackend> &, AccessOrder,
                                                       bool);
template void TensorList<GPUBackend>::Copy<GPUBackend>(const TensorList<GPUBackend> &, AccessOrder,
                                                       bool);

}  // namespace dali
