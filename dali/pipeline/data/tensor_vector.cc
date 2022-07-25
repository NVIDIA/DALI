// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/data/tensor_vector.h"
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

  // Wait on the order on which we will run the copy for the work to finish on the dst
  order.wait(dst_order);

  return order;
}


/**
 * @brief Wait for the reallocation to happen in the copy order, so we can actually proceed.
 */
void SyncAfterResize(AccessOrder dst_order, AccessOrder copy_order) {
  copy_order.wait(dst_order);
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
  SmallVector<const void *, kMaxStaticCopyBatchSize> srcs;
  srcs.reserve(num_samples);
  SmallVector<void *, kMaxStaticCopyBatchSize> dsts;
  dsts.reserve(num_samples);
  SmallVector<Index, kMaxStaticCopyBatchSize> sizes;
  sizes.reserve(num_samples);
  for (int i = 0; i < num_samples; i++) {
    dsts.emplace_back(dst.raw_mutable_tensor(i));
    srcs.emplace_back(src.raw_tensor(i));
    sizes.emplace_back(src.shape()[i].num_elements());
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
  SmallVector<void *, kMaxStaticCopyBatchSize> dsts;
  dsts.reserve(num_samples);
  SmallVector<Index, kMaxStaticCopyBatchSize> sizes;
  sizes.reserve(num_samples);
  for (int i = 0; i < num_samples; i++) {
    dsts.emplace_back(dst.raw_mutable_tensor(i));
    sizes.emplace_back(dst.shape()[i].num_elements());
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
  SmallVector<const void *, kMaxStaticCopyBatchSize> srcs;
  srcs.reserve(num_samples);
  SmallVector<Index, kMaxStaticCopyBatchSize> sizes;
  sizes.reserve(num_samples);
  for (int i = 0; i < num_samples; i++) {
    srcs.emplace_back(src.raw_tensor(i));
    sizes.emplace_back(src.shape()[i].num_elements());
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
    type_info.Copy<DstBackend, SrcBackend>(unsafe_raw_mutable_data(dst), unsafe_raw_data(src),
                                           dst.shape().num_elements(), copy_order.stream(),
                                           use_copy_kernel);
  } else if (dst.IsContiguous() && !src.IsContiguous()) {
    copy_impl::CopySamplewiseImpl<DstBackend, SrcBackend>(unsafe_raw_mutable_data(dst), src,
                                                          type_info, copy_order, use_copy_kernel);
  } else if (!dst.IsContiguous() && src.IsContiguous()) {
    copy_impl::CopySamplewiseImpl<DstBackend, SrcBackend>(dst, unsafe_raw_data(src), type_info,
                                                          copy_order, use_copy_kernel);
  } else {
    copy_impl::CopySamplewiseImpl<DstBackend, SrcBackend>(dst, src, type_info, copy_order,
                                                          use_copy_kernel);
  }
}

/** @} */  // end of copy_impl

}  // namespace copy_impl

template <typename Backend>
TensorVector<Backend>::TensorVector()
    : views_count_(0), curr_num_tensors_(0), tl_(std::make_shared<TensorList<Backend>>()) {}


template <typename Backend>
TensorVector<Backend>::TensorVector(int batch_size)
    : views_count_(0),
      curr_num_tensors_(0),
      tl_(std::make_shared<TensorList<Backend>>(batch_size)) {
  resize_tensors(batch_size);
}

template <typename Backend>
TensorVector<Backend>::TensorVector(TensorVector<Backend> &&other) noexcept {
  state_ = other.state_;
  pinned_ = other.pinned_;
  order_ = other.order_;
  curr_num_tensors_ = other.curr_num_tensors_;
  tl_ = std::move(other.tl_);
  type_ = std::move(other.type_);
  sample_dim_ = other.sample_dim_;
  views_count_ = other.views_count_.load();
  tensors_ = std::move(other.tensors_);
  for (auto &t : tensors_) {
    if (t) {
      if (auto *del = std::get_deleter<ViewRefDeleter>(t->data_)) del->ref = &views_count_;
    }
  }

  other.views_count_ = 0;
  other.curr_num_tensors_ = 0;
  other.tensors_.clear();
  other.sample_dim_ = -1;
}

template <typename Backend>
void TensorVector<Backend>::UnsafeSetSample(int sample_idx, const TensorVector<Backend> &src,
                                            int src_sample_idx) {
  // TODO(klecki): more consistency checks, contiguous -> non-contiguous removes shares_data from
  // samples
  // Bounds check
  assert(sample_idx >= 0 && sample_idx < curr_num_tensors_);
  assert(src_sample_idx >= 0 && src_sample_idx < src.curr_num_tensors_);
  DALI_ENFORCE(type() == src.type(),
               make_string("Sample must have the same type as a target batch, current: ", type(),
                           " new: ", src.type(), " for ", sample_idx, " <- ", src_sample_idx, "."));
  DALI_ENFORCE(sample_dim() == src.shape().sample_dim(),
               make_string("Sample must have the same dimensionality as a target batch, current: ",
                           sample_dim(), " new: ", src.shape().sample_dim(), " for ", sample_idx,
                           " <- ", src_sample_idx, "."));
  DALI_ENFORCE(this->order() == src.order(), "Sample must have the same order as a target batch");
  DALI_ENFORCE(
      GetLayout() == src.GetLayout(),
      make_string("Sample must have the same layout as a target batch current: ", GetLayout(),
                  " new: ", src.GetLayout(), " for ", sample_idx, " <- ", src_sample_idx, "."));
  DALI_ENFORCE(
      is_pinned() == src.is_pinned(),
      make_string("Sample must have the same pinned status as target batch, current: ", is_pinned(),
                  " new: ", src.is_pinned(), " for ", sample_idx, " <- ", src_sample_idx, "."));

  SetContiguous(false);
  // Setting a new share overwrites the previous one - so we can safely assume that even if
  // we had a sample sharing into TL, it will be overwritten
  tensors_[sample_idx]->ShareData(*src.tensors_[src_sample_idx]);
  tl_->Reset();
}

template <typename Backend>
void TensorVector<Backend>::UnsafeSetSample(int sample_idx, const Tensor<Backend> &owner) {
  // TODO(klecki): more consistency checks, contiguous -> non-contiguous removes shares_data from
  // samples
  // Bounds check
  assert(sample_idx >= 0 && sample_idx < curr_num_tensors_);
  DALI_ENFORCE(type() == owner.type(),
               make_string("Sample must have the same type as a target batch, current: ", type(),
                           " new: ", owner.type(), " for ", sample_idx, "."));
  DALI_ENFORCE(
      sample_dim() == owner.shape().sample_dim(),
      make_string("Sample must have the same dimensionality as a target batch, current: ",
                  sample_dim(), " new: ", owner.shape().sample_dim(), " for ", sample_idx, "."));
  DALI_ENFORCE(this->order() == owner.order(), "Sample must have the same order as a target batch");
  DALI_ENFORCE(GetLayout() == owner.GetLayout(),
               make_string("Sample must have the same layout as a target batch current: ",
                           GetLayout(), " new: ", owner.GetLayout(), " for ", sample_idx, "."));
  DALI_ENFORCE(
      is_pinned() == owner.is_pinned(),
      make_string("Sample must have the same pinned status as target batch, current: ", is_pinned(),
                  " new: ", owner.is_pinned(), " for ", sample_idx, "."));
  SetContiguous(false);
  // Setting a new share overwrites the previous one - so we can safely assume that even if
  // we had a sample sharing into TL, it will be overwritten
  tensors_[sample_idx]->ShareData(owner);
  tl_->Reset();
}

template <typename Backend>
void TensorVector<Backend>::UnsafeSetSample(int sample_idx, const shared_ptr<void> &ptr,
                                            size_t bytes, bool pinned, const TensorShape<> &shape,
                                            DALIDataType type, int device_id, AccessOrder order,
                                            const TensorLayout &layout) {
  assert(sample_idx >= 0 && sample_idx < curr_num_tensors_);
  DALI_ENFORCE(this->type() == type,
               make_string("Sample must have the same type as a target batch, current: ",
                           this->type(), " new: ", type, " for ", sample_idx, "."));
  DALI_ENFORCE(sample_dim() == shape.sample_dim(),
               make_string("Sample must have the same dimensionality as a target batch, current: ",
                           sample_dim(), " new: ", shape.sample_dim(), " for ", sample_idx, "."));
  DALI_ENFORCE(this->device_id() == device_id,
               make_string("Sample must have the same device id as a target batch, current: ",
                           this->device_id(), " new: ", device_id, " for ", sample_idx, "."));
  DALI_ENFORCE(this->order() == order, "Sample must have the same order as a target batch");
  DALI_ENFORCE(GetLayout() == layout,
               make_string("Sample must have the same layout as a target batch current: ",
                           GetLayout(), " new: ", layout, " for ", sample_idx, "."));
  DALI_ENFORCE(is_pinned() == pinned,
               make_string("Sample must have the same pinned status as target batch, current: ",
                           is_pinned(), " new: ", pinned, " for ", sample_idx, "."));
  SetContiguous(false);
  // Setting a new share overwrites the previous one - so we can safely assume that even if
  // we had a sample sharing into TL, it will be overwritten
  tensors_[sample_idx]->ShareData(ptr, bytes, pinned, shape, type, device_id, order);
  tl_->Reset();
}

template <typename Backend>
void TensorVector<Backend>::UnsafeCopySample(int sample_idx, const TensorVector<Backend> &src,
                                             int src_sample_idx, AccessOrder order) {
  // TODO(klecki): more consistency checks, contiguous -> non-contiguous removes shares_data from
  // samples
  // Bounds check
  assert(sample_idx >= 0 && sample_idx < curr_num_tensors_);
  assert(src_sample_idx >= 0 && src_sample_idx < src.curr_num_tensors_);
  DALI_ENFORCE(type() == src.type(),
               make_string("Sample must have the same type as a target batch, current: ", type(),
                           " new: ", src.type(), " for ", sample_idx, " <- ", src_sample_idx, "."));
  DALI_ENFORCE(sample_dim() == src.shape().sample_dim(),
               make_string("Sample must have the same dimensionality as a target batch, current: ",
                           sample_dim(), " new: ", src.shape().sample_dim(), " for ", sample_idx,
                           " <- ", src_sample_idx, "."));
  DALI_ENFORCE(
      GetLayout() == src.GetLayout(),
      make_string("Sample must have the same layout as a target batch current: ", GetLayout(),
                  " new: ", src.GetLayout(), " for ", sample_idx, " <- ", src_sample_idx, "."));

  // Either the shape matches and we can copy data as is or the target is just an individual sample
  bool can_copy = tensors_[sample_idx]->shape() == src.tensors_[src_sample_idx]->shape() ||
                  (!tl_->has_data() && state_ == State::noncontiguous);

  DALI_ENFORCE(
      can_copy,
      "Copying samples into TensorVector can happen either for exact shape match or when the "
      "TensorVector is truly non contiguous. Either Resize first to the desired shape or reset the "
      "TensorVector and SetSize for desired number of samples in non-contiguous mode.");

  tensors_[sample_idx]->Copy(*src.tensors_[src_sample_idx], order);
}

template <typename Backend>
void TensorVector<Backend>::set_sample_dim(int sample_dim) {
  DALI_ENFORCE(
      !has_data(),
      "Setting sample dim is not allowed when batch is already allocated, use Resize instead.");
  sample_dim_ = sample_dim;
}

template <typename Backend>
size_t TensorVector<Backend>::nbytes() const noexcept {
  if (state_ == State::contiguous) {
    return tl_->nbytes();
  }
  // else
  size_t nbytes = 0;
  for (const auto &t : tensors_) {
    nbytes += t->nbytes();
  }
  return nbytes;
}


template <typename Backend>
size_t TensorVector<Backend>::capacity() const noexcept {
  if (state_ == State::contiguous) {
    return tl_->capacity();
  }
  // else
  size_t capacity = 0;
  for (const auto &t : tensors_) {
    capacity += t->capacity();
  }
  return capacity;
}

template <typename Backend>
std::vector<size_t> TensorVector<Backend>::_chunks_nbytes() const {
  if (state_ == State::contiguous) {
    return {tl_->nbytes()};
  }
  // else
  std::vector<size_t> result(tensors_.size());
  for (size_t i = 0; i < tensors_.size(); i++) {
    result[i] = tensors_[i]->nbytes();
  }
  return result;
}


template <typename Backend>
std::vector<size_t> TensorVector<Backend>::_chunks_capacity() const {
  if (state_ == State::contiguous) {
    return {tl_->capacity()};
  }
  // else
  std::vector<size_t> result(tensors_.size());
  for (size_t i = 0; i < tensors_.size(); i++) {
    result[i] = tensors_[i]->capacity();
  }
  return result;
}


template <typename Backend>
TensorListShape<> TensorVector<Backend>::shape() const {
  if (state_ == State::contiguous) {
    return tl_->shape();
  }
  if (curr_num_tensors_ == 0) {
    return {};
  }
  TensorListShape<> result(curr_num_tensors_, tensors_[0]->ndim());
  for (int i = 0; i < curr_num_tensors_; i++) {
    result.set_tensor_shape(i, tensors_[i]->shape());
  }
  return result;
}

template <typename Backend>
void TensorVector<Backend>::set_order(AccessOrder order, bool synchronize) {
  // Optimization: synchronize only once, if needed.
  if (this->order().is_device() && order && synchronize) {
    bool need_sync = tl_->has_data();
    if (!need_sync) {
      for (auto &t : tensors_) {
        if (t->has_data()) {
          need_sync = true;
          break;
        }
      }
    }
    if (need_sync)
      this->order().wait(order);
  }
  tl_->set_order(order, false);
  for (auto &t : tensors_)
    t->set_order(order, false);
  order_ = order;
}

template <typename Backend>
void TensorVector<Backend>::Resize(const TensorListShape<> &new_shape, DALIDataType new_type) {
  DALI_ENFORCE(IsValidType(new_type),
                "TensorVector cannot be resized with invalid type. To zero out the TensorVector "
                "Reset() can be used.");
  resize_tensors(new_shape.num_samples());
  if (state_ == State::contiguous) {
    tl_->Resize(new_shape, new_type);
    UpdateViews();
    return;
  }

  for (int i = 0; i < curr_num_tensors_; i++) {
    tensors_[i]->Resize(new_shape[i], new_type);
  }
  if (type_.id() != new_type) {
    type_ = TypeTable::GetTypeInfo(new_type);
    if (state_ == State::noncontiguous) {
      tl_->Reset();
      tl_->set_type(new_type);
    }
  }
  sample_dim_ = new_shape.sample_dim();
}


template <typename Backend>
void TensorVector<Backend>::SetSize(int new_size) {
  DALI_ENFORCE(new_size >= 0, make_string("Incorrect size: ", new_size));
  resize_tensors(new_size);
}


template <typename Backend>
void TensorVector<Backend>::set_type(DALIDataType new_type_id) {
  DALI_ENFORCE(new_type_id != DALI_NO_TYPE, "new_type must be valid type.");
  if (type_.id() == new_type_id)
    return;
  DALI_ENFORCE(type_.id() == new_type_id || (!has_data() || type_.id() == DALI_NO_TYPE),
               make_string("set_type cannot be used to change the current type - it is not "
                           "allowed to cause allocations. Currently set type: '",
                           type_.id(), "' trying to set: '", new_type_id,
                           "'. You may change the current type using Resize or by"
                           " calling Reset first."));
  type_ = TypeTable::GetTypeInfo(new_type_id);
  tl_->set_type(new_type_id);
  for (auto t : tensors_) {
    t->set_type(new_type_id);
  }
  if (state_ == State::contiguous) {
    UpdateViews();
  }
}


template <typename Backend>
DALIDataType TensorVector<Backend>::type() const {
  if (state_ == State::contiguous) {
    return tl_->type();
  }
  if (curr_num_tensors_ == 0) {
    return type_.id();
  }
  for (int i = 1; i < curr_num_tensors_; i++) {
    assert(tensors_[0]->type() == tensors_[i]->type());
  }
  return tensors_[0]->type();
}

template <typename Backend>
const TypeInfo &TensorVector<Backend>::type_info() const {
  if (state_ == State::contiguous) {
    return tl_->type_info();
  }
  if (curr_num_tensors_ == 0) {
    return type_;
  }
  for (int i = 1; i < curr_num_tensors_; i++) {
    assert(tensors_[0]->type() == tensors_[i]->type());
  }
  return tensors_[0]->type_info();
}


template <typename Backend>
void TensorVector<Backend>::SetLayout(const TensorLayout &layout) {
  if (state_ == State::noncontiguous) {
    DALI_ENFORCE(!tensors_.empty(), "Layout cannot be set uniformly for empty batch");
  }
  tl_->SetLayout(layout);
  for (auto t : tensors_) {
    t->SetLayout(layout);
  }
}


template <typename Backend>
void TensorVector<Backend>::SetSkipSample(int idx, bool skip_sample) {
  tensors_[idx]->SetSkipSample(skip_sample);
}


template <typename Backend>
void TensorVector<Backend>::SetSourceInfo(int idx, const std::string& source_info) {
  tensors_[idx]->SetSourceInfo(source_info);
}


template <typename Backend>
TensorLayout TensorVector<Backend>::GetLayout() const {
  if (state_ == State::contiguous) {
    auto layout = tl_->GetLayout();
    if (!layout.empty()) return layout;
  }
  if (curr_num_tensors_ > 0) {
    auto layout = tensors_[0]->GetLayout();
    for (int i = 1; i < curr_num_tensors_; i++) assert(layout == tensors_[i]->GetLayout());
    return layout;
  }
  return {};
}


template <typename Backend>
const DALIMeta &TensorVector<Backend>::GetMeta(int idx) const {
  assert(idx < curr_num_tensors_);
  return tensors_[idx]->GetMeta();
}


template <typename Backend>
void TensorVector<Backend>::SetMeta(int idx, const DALIMeta &meta) {
  assert(idx < curr_num_tensors_);
  tensors_[idx]->SetMeta(meta);
}


template <typename Backend>
void TensorVector<Backend>::set_pinned(bool pinned) {
  // Store the value, in case we pin empty vector and later call Resize
  pinned_ = pinned;
  tl_->set_pinned(pinned);
  for (auto &t : tensors_) {
    t->set_pinned(pinned);
  }
}


template <typename Backend>
bool TensorVector<Backend>::is_pinned() const {
  return pinned_;
}


template <typename Backend>
void TensorVector<Backend>::set_device_id(int device_id) {
  tl_->set_device_id(device_id);
  for (auto &t : tensors_) {
    t->set_device_id(device_id);
  }
}


template <typename Backend>
int TensorVector<Backend>::device_id() const {
  if (IsContiguous()) {
    return tl_->device_id();
  } else if (!tensors_.empty()) {
    return tensors_[0]->device_id();
  }
  return CPU_ONLY_DEVICE_ID;
}


template <typename Backend>
void TensorVector<Backend>::reserve(size_t total_bytes) {
  if (state_ == State::noncontiguous) {
    tensors_.clear();
    curr_num_tensors_ = 0;
  }
  state_ = State::contiguous;
  tl_->reserve(total_bytes);
  UpdateViews();
}


template <typename Backend>
void TensorVector<Backend>::reserve(size_t bytes_per_sample, int batch_size) {
  assert(batch_size > 0);
  state_ = State::noncontiguous;
  resize_tensors(batch_size);
  for (int i = 0; i < curr_num_tensors_; i++) {
    tensors_[i]->reserve(bytes_per_sample);
  }
}


template <typename Backend>
bool TensorVector<Backend>::IsContiguous() const noexcept {
  return state_ == State::contiguous && views_count_ == num_samples();
}


template <typename Backend>
void TensorVector<Backend>::SetContiguous(bool contiguous) {
  if (contiguous) {
    state_ = State::contiguous;
  } else {
    state_ = State::noncontiguous;
  }
}


template <typename Backend>
void TensorVector<Backend>::Reset() {
  tensors_.clear();
  curr_num_tensors_ = 0;
  type_ = {};
  sample_dim_ = -1;
  if (IsContiguous()) {
    views_count_ = 0;
    tl_->Reset();
  }
}


template <typename Backend>
template <typename SrcBackend>
void TensorVector<Backend>::Copy(const TensorList<SrcBackend> &in_tl, AccessOrder order) {
  // This variant will be removed with the removal of TensorList.
  SetContiguous(true);

  auto copy_order = copy_impl::SyncBefore(this->order(), in_tl.order(), order);


  tl_->Resize(in_tl.shape(), in_tl.type());
  type_ = in_tl.type_info();
  sample_dim_ = in_tl.shape().sample_dim();

  copy_impl::SyncAfterResize(this->order(), copy_order);

  // Update the metadata
  type_ = in_tl.type_info();
  sample_dim_ = in_tl.shape().sample_dim();

  // Here both batches are contiguous
  type_info().template Copy<Backend, SrcBackend>(
      unsafe_raw_mutable_data(*tl_), unsafe_raw_data(in_tl), in_tl.shape().num_elements(),
      copy_order.stream(), false);
  copy_impl::SyncAfter(this->order(), copy_order);

  resize_tensors(tl_->num_samples());
  UpdateViews();

  // Update the layout and other metadata
  SetLayout(in_tl.GetLayout());
  for (int i = 0; i < curr_num_tensors_; i++) {
    SetMeta(i, in_tl.GetMeta(i));
  }
}


template <typename Backend>
template <typename SrcBackend>
void TensorVector<Backend>::Copy(const TensorVector<SrcBackend> &in_tv, AccessOrder order) {
  auto copy_order = copy_impl::SyncBefore(this->order(), in_tv.order(), order);

  Resize(in_tv.shape(), in_tv.type());
  // After resize the state_, curr_num_tensors_, type_, sample_dim_, shape_ (and pinned)
  // postconditions are met, as well as the buffers are correctly adjusted.
  copy_impl::SyncAfterResize(this->order(), copy_order);

  bool use_copy_kernel = false;

  copy_impl::CopyImpl(*this, in_tv, this->type_info(), copy_order, use_copy_kernel);

  // Update the layout and other metadata
  SetLayout(in_tv.GetLayout());
  for (int i = 0; i < curr_num_tensors_; i++) {
    SetMeta(i, in_tv.GetMeta(i));
  }
  copy_impl::SyncAfter(this->order(), copy_order);
}


template <typename Backend>
void TensorVector<Backend>::ShareData(const TensorList<Backend> &in_tl) {
  SetContiguous(true);
  type_ = in_tl.type_info();
  sample_dim_ = in_tl.shape().sample_dim();
  pinned_ = in_tl.is_pinned();
  tl_->ShareData(in_tl);

  resize_tensors(in_tl.num_samples());
  UpdateViews();
}

template <typename Backend>
void TensorVector<Backend>::ShareData(const TensorVector<Backend> &tv) {
  type_ = tv.type_;
  sample_dim_ = tv.sample_dim_;
  state_ = tv.state_;
  pinned_ = tv.is_pinned();
  views_count_ = 0;
  if (tv.state_ == State::contiguous) {
    ShareData(*tv.tl_);
  } else {
    state_ = State::noncontiguous;
    tl_->Reset();
    int batch_size = tv.num_samples();
    for (int i = 0; i < batch_size; i++) {
      resize_tensors(batch_size);
      tensors_[i]->ShareData(*(tv.tensors_[i]));
    }
  }
}


template <typename Backend>
TensorVector<Backend> &TensorVector<Backend>::operator=(TensorVector<Backend> &&other) noexcept {
  if (&other != this) {
    state_ = other.state_;
    pinned_ = other.pinned_;
    order_ = other.order_;
    curr_num_tensors_ = other.curr_num_tensors_;
    tl_ = std::move(other.tl_);
    type_ = other.type_;
    sample_dim_ = other.sample_dim_;
    views_count_ = other.views_count_.load();
    tensors_ = std::move(other.tensors_);
    for (auto &t : tensors_) {
      if (t) {
        if (auto *del = std::get_deleter<ViewRefDeleter>(t->data_)) del->ref = &views_count_;
      }
    }

    other.views_count_ = 0;
    other.curr_num_tensors_ = 0;
    other.tensors_.clear();
  }
  return *this;
}


template <typename Backend>
void TensorVector<Backend>::UpdateViews() {
  // Return if we do not have a valid allocation
  if (!IsValidType(tl_->type())) return;
  // we need to be able to share empty view as well so don't check if tl_ has any data
  type_ = tl_->type_info();
  sample_dim_ = tl_->shape().sample_dim();

  assert(curr_num_tensors_ == tl_->num_samples());

  views_count_ = curr_num_tensors_;
  for (int i = 0; i < curr_num_tensors_; i++) {
    update_view(i);
  }
}


template <typename Backend>
void TensorVector<Backend>::resize_tensors(int new_size) {
  if (static_cast<size_t>(new_size) > tensors_.size()) {
    auto old_size = curr_num_tensors_;
    tensors_.resize(new_size);
    for (int i = old_size; i < new_size; i++) {
      if (!tensors_[i]) {
        tensors_[i] = std::make_shared<Tensor<Backend>>();
        tensors_[i]->set_pinned(is_pinned());
        tensors_[i]->set_order(order());
      }
    }
  } else if (new_size < curr_num_tensors_) {
    for (int i = new_size; i < curr_num_tensors_; i++) {
      if (tensors_[i]->shares_data()) {
        tensors_[i]->Reset();
      }
    }
    // TODO(klecki): Do not keep the invalidated tensors - this prevents memory hogging but
    // also gets rid of reserved memory.
    // tensors_.resize(new_size);
  }
  curr_num_tensors_ = new_size;
}

template <typename Backend>
void TensorVector<Backend>::UpdatePropertiesFromSamples(bool contiguous) {
  // TODO(klecki): This is mostly simple consistency check, but most of the metadata will be moved
  // to the batch object for consitency and easier use in checks. It should allow for shape()
  // to be ready to use as well as easy verification for SetSample/CopySample.
  SetContiguous(contiguous);
  // assume that the curr_num_tensors_ is valid
  DALI_ENFORCE(curr_num_tensors_ > 0, "Unexpected empty output of operator. Internal DALI error.");
  type_ = tensors_[0]->type_info();
  sample_dim_ = tensors_[0]->shape().sample_dim();
  pinned_ = tensors_[0]->is_pinned();
  order_ = tensors_[0]->order();
  tl_->set_order(order_);
  for (int i = 0; i < curr_num_tensors_; i++) {
    DALI_ENFORCE(type() == tensors_[i]->type(),
                 make_string("Samples must have the same type, expected: ", type(),
                             " got: ", tensors_[i]->type(), " at ", i, "."));
    DALI_ENFORCE(sample_dim() == tensors_[i]->shape().sample_dim(),
                 make_string("Samples must have the same dimensionality, expected: ", sample_dim(),
                             " got: ", tensors_[i]->shape().sample_dim(), " at ", i, "."));
    DALI_ENFORCE(order() == tensors_[i]->order(),
                 make_string("Samples must have the same order, expected: ", order().get(), " ",
                             order().device_id(), " got: ", tensors_[i]->order().get(), " ",
                             tensors_[i]->order().device_id(), " at ", i, "."));
    DALI_ENFORCE(GetLayout() == tensors_[i]->GetLayout(),
                 make_string("Samples must have the same layout, expected: ", GetLayout(),
                             " got: ", tensors_[i]->GetLayout(), " at ", i, "."));
  }
}

template <typename Backend>
bool TensorVector<Backend>::has_data() const {
  if (state_ == State::contiguous) {
    return tl_->has_data();
  }
  for (const auto &tensor : tensors_) {
    if (tensor->has_data()) {
      return true;
    }
  }
  return false;
}

template <typename Backend>
void TensorVector<Backend>::update_view(int idx) {
  assert(idx < curr_num_tensors_);
  assert(idx < tl_->num_samples());

  auto *ptr = tl_->raw_mutable_tensor(idx);

  TensorShape<> shape = tl_->tensor_shape(idx);

  tensors_[idx]->Reset();
  // TODO(klecki): deleter that reduces views_count or just noop sharing?
  // tensors_[i]->ShareData(tl_.get(), static_cast<int>(idx));
  if (tensors_[idx]->raw_data() != ptr || tensors_[idx]->shape() != shape) {
    tensors_[idx]->ShareData(std::shared_ptr<void>(ptr, ViewRefDeleter{&views_count_}),
                             volume(tl_->tensor_shape(idx)) * tl_->type_info().size(),
                             tl_->is_pinned(), shape, tl_->type(), tl_->device_id(), order());
  } else if (IsValidType(tl_->type())) {
    tensors_[idx]->set_type(tl_->type());
  }
  tensors_[idx]->SetMeta(tl_->GetMeta(idx));
}


template class DLL_PUBLIC TensorVector<CPUBackend>;
template class DLL_PUBLIC TensorVector<GPUBackend>;
template void TensorVector<CPUBackend>::Copy<CPUBackend>(const TensorVector<CPUBackend>&, AccessOrder);  // NOLINT
template void TensorVector<CPUBackend>::Copy<GPUBackend>(const TensorVector<GPUBackend>&, AccessOrder);  // NOLINT
template void TensorVector<GPUBackend>::Copy<CPUBackend>(const TensorVector<CPUBackend>&, AccessOrder);  // NOLINT
template void TensorVector<GPUBackend>::Copy<GPUBackend>(const TensorVector<GPUBackend>&, AccessOrder);  // NOLINT
template void TensorVector<CPUBackend>::Copy<CPUBackend>(const TensorList<CPUBackend>&, AccessOrder);  // NOLINT
template void TensorVector<CPUBackend>::Copy<GPUBackend>(const TensorList<GPUBackend>&, AccessOrder);  // NOLINT
template void TensorVector<GPUBackend>::Copy<CPUBackend>(const TensorList<CPUBackend>&, AccessOrder);  // NOLINT
template void TensorVector<GPUBackend>::Copy<GPUBackend>(const TensorList<GPUBackend>&, AccessOrder);  // NOLINT

}  // namespace dali
