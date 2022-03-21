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

#include "dali/pipeline/data/tensor_vector.h"
#include "dali/core/common.h"

namespace dali {

template <typename Backend>
TensorVector<Backend>::TensorVector()
    : views_count_(0), curr_tensors_size_(0), tl_(std::make_shared<TensorList<Backend>>()) {}


template <typename Backend>
TensorVector<Backend>::TensorVector(int batch_size)
    : views_count_(0),
      curr_tensors_size_(0),
      tl_(std::make_shared<TensorList<Backend>>(batch_size)) {
  resize_tensors(batch_size);
}


template <typename Backend>
TensorVector<Backend>::TensorVector(std::shared_ptr<TensorList<Backend>> tl)
    : views_count_(0), curr_tensors_size_(0), tl_(std::move(tl)) {
  assert(tl_ && "Construction with null TensorList is illegal");
  pinned_ = tl_->is_pinned();
  type_ = tl_->type_info();
  state_ = State::contiguous;
  resize_tensors(tl_->num_samples());
  UpdateViews();
}


template <typename Backend>
TensorVector<Backend>::TensorVector(TensorVector<Backend> &&other) noexcept {
  state_ = other.state_;
  pinned_ = other.pinned_;
  curr_tensors_size_ = other.curr_tensors_size_;
  tl_ = std::move(other.tl_);
  type_ = std::move(other.type_);
  views_count_ = other.views_count_.load();
  tensors_ = std::move(other.tensors_);
  for (auto &t : tensors_) {
    if (t) {
      if (auto *del = std::get_deleter<ViewRefDeleter>(t->data_)) del->ref = &views_count_;
    }
  }

  other.views_count_ = 0;
  other.curr_tensors_size_ = 0;
  other.tensors_.clear();
}

template <typename Backend>
void TensorVector<Backend>::UnsafeSetSample(int dst, const TensorVector<Backend> &owner, int src) {
  // TODO(klecki): more consistency checks, contiguous -> non-contiguous removes shares_data from
  // samples
  if (type() == DALI_NO_TYPE && owner.type() != DALI_NO_TYPE) {
    set_type(owner.type());
  }
  if (!order()) {
    set_order(owner.order());
  }
  // Bounds check
  assert(dst >= 0 && dst < static_cast<int>(curr_tensors_size_));
  assert(src >= 0 && src < static_cast<int>(owner.curr_tensors_size_));
  DALI_ENFORCE(type() == owner.type(),
               make_string("Sample must have the same type as a target batch, current: ", type(),
                           " new: ", owner.type(), " for ", dst, " <- ", src, "."));
  DALI_ENFORCE(tensor_shape(dst) == TensorShape<>{0} || sample_dim() == owner.shape().sample_dim(),
               make_string("Sample must have the same dimensionality as a target batch, current: ",
                           sample_dim(), " new: ", owner.shape().sample_dim(), " for ", dst, " <- ",
                           src, "."));
  DALI_ENFORCE(this->order() == owner.order(), "Sample must have the same order as a target batch");
  DALI_ENFORCE(
      GetLayout() == "" || GetLayout() == owner.GetLayout(),
      make_string("Sample must have the same layout as a target batch current: ", GetLayout(),
                  " new: ", owner.GetLayout(), " for ", dst, " <- ", src, "."));
  DALI_ENFORCE(
      is_pinned() == owner.is_pinned(),
      make_string("Sample must have the same pinned status as target batch, current: ", is_pinned(),
                  " new: ", owner.is_pinned(), " for ", dst, " <- ", src, "."));

  SetContiguous(false);
  // Setting a new share overwrites the previous one - so we can safely assume that even if
  // we had a sample sharing into TL, it will be overwritten
  tensors_[dst]->ShareData(*owner.tensors_[src]);
  tl_->Reset();
}

template <typename Backend>
void TensorVector<Backend>::UnsafeSetSample(int dst, const Tensor<Backend> &owner) {
  // TODO(klecki): more consistency checks, contiguous -> non-contiguous removes shares_data from
  // samples
  if (type() == DALI_NO_TYPE && owner.type() != DALI_NO_TYPE) {
    set_type(owner.type());
  }
  if (!order()) {
    set_order(owner.order());
  }
  // Bounds check
  assert(dst >= 0 && dst < static_cast<int>(curr_tensors_size_));
  DALI_ENFORCE(type() == owner.type(),
               make_string("Sample must have the same type as a target batch, current: ", type(),
                           " new: ", owner.type(), " for ", dst, " <-."));
  DALI_ENFORCE(
      tensor_shape(dst) == TensorShape<>{0} || sample_dim() == owner.shape().sample_dim(),
      make_string("Sample must have the same dimensionality as a target batch, current: ",
                  sample_dim(), " new: ", owner.shape().sample_dim(), " for ", dst, " <-."));
  DALI_ENFORCE(this->order() == owner.order(), "Sample must have the same order as a target batch");
  DALI_ENFORCE(
      GetLayout() == "" || GetLayout() == owner.GetLayout(),
      make_string("Sample must have the same layout as a target batch current: ", GetLayout(),
                  " new: ", owner.GetLayout(), " for ", dst, " <-."));
  DALI_ENFORCE(
      is_pinned() == owner.is_pinned(),
      make_string("Sample must have the same pinned status as target batch, current: ", is_pinned(),
                  " new: ", owner.is_pinned(), " for ", dst, " <-."));
  SetContiguous(false);
  // Setting a new share overwrites the previous one - so we can safely assume that even if
  // we had a sample sharing into TL, it will be overwritten
  tensors_[dst]->ShareData(owner);
  tl_->Reset();
}

template <typename Backend>
void TensorVector<Backend>::UnsafeCopySample(int dst, const TensorVector<Backend> &data, int src,
                                             AccessOrder order) {
  // TODO(klecki): more consistency checks, contiguous -> non-contiguous removes shares_data from
  // samples
  if (type() == DALI_NO_TYPE && data.type() != DALI_NO_TYPE) {
    set_type(data.type());
  }
  // Bounds check
  assert(dst >= 0 && dst < static_cast<int>(curr_tensors_size_));
  assert(src >= 0 && src < static_cast<int>(data.curr_tensors_size_));
  DALI_ENFORCE(type() == data.type(),
               make_string("Sample must have the same type as a target batch, current: ", type(),
                           " new: ", data.type(), " for ", dst, " <- ", src, "."));
  DALI_ENFORCE(tensor_shape(dst) == TensorShape<>{0} || sample_dim() == data.shape().sample_dim(),
               make_string("Sample must have the same dimensionality as a target batch, current: ",
                           sample_dim(), " new: ", data.shape().sample_dim(), " for ", dst, " <- ",
                           src, "."));
  DALI_ENFORCE(
      GetLayout() == "" || GetLayout() == data.GetLayout(),
      make_string("Sample must have the same layout as a target batch current: ", GetLayout(),
                  " new: ", data.GetLayout(), " for ", dst, " <- ", src, "."));

  // Either the shape matches and we can copy data as is or the target is just an individual sample
  bool can_copy =
      tensors_[dst]->shape() == data.tensors_[src]->shape() ||
      (!tl_->has_data() && state_ == State::noncontiguous);

  DALI_ENFORCE(
      can_copy,
      "Copying samples into TensorVector can happen either for exact shape match or when the "
      "TensorVector is truly non contiguous. Either Resize first to the desired shape or reset the "
      "TensorVector and SetSize for desired number of samples in non-contiguous mode.");

  tensors_[dst]->Copy(*data.tensors_[src], order);
}


template <typename Backend>
size_t TensorVector<Backend>::total_nbytes() const noexcept {
  if (state_ == State::contiguous) {
    return tl_->total_nbytes();
  }
  // else
  size_t total_nbytes = 0;
  for (const auto &t : tensors_) {
    total_nbytes += t->nbytes();
  }
  return total_nbytes;
}


template <typename Backend>
size_t TensorVector<Backend>::total_capacity() const noexcept {
  if (state_ == State::contiguous) {
    return tl_->total_capacity();
  }
  // else
  size_t total_capacity = 0;
  for (const auto &t : tensors_) {
    total_capacity += t->capacity();
  }
  return total_capacity;
}

template <typename Backend>
std::vector<size_t> TensorVector<Backend>::nbytes() const noexcept {
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
std::vector<size_t> TensorVector<Backend>::capacity() const noexcept {
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
  if (curr_tensors_size_ == 0) {
    return {};
  }
  TensorListShape<> result(curr_tensors_size_, tensors_[0]->ndim());
  for (size_t i = 0; i < curr_tensors_size_; i++) {
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

  for (size_t i = 0; i < curr_tensors_size_; i++) {
    tensors_[i]->Resize(new_shape[i], new_type);
  }
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
  if (curr_tensors_size_ == 0) {
    return type_.id();
  }
  for (size_t i = 1; i < curr_tensors_size_; i++) {
    assert(tensors_[0]->type() == tensors_[i]->type());
  }
  return tensors_[0]->type();
}

template <typename Backend>
const TypeInfo &TensorVector<Backend>::type_info() const {
  if (state_ == State::contiguous) {
    return tl_->type_info();
  }
  if (curr_tensors_size_ == 0) {
    return type_;
  }
  for (size_t i = 1; i < curr_tensors_size_; i++) {
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
TensorLayout TensorVector<Backend>::GetLayout() const {
  if (state_ == State::contiguous) {
    auto layout = tl_->GetLayout();
    if (!layout.empty()) return layout;
  }
  if (curr_tensors_size_ > 0) {
    auto layout = tensors_[0]->GetLayout();
    for (size_t i = 1; i < curr_tensors_size_; i++) assert(layout == tensors_[i]->GetLayout());
    return layout;
  }
  return {};
}

template <typename Backend>
DALIMeta &TensorVector<Backend>::GetMeta(int idx) {
  assert(static_cast<size_t>(idx) < curr_tensors_size_);
  return tensors_[idx]->GetMeta();
}

template <typename Backend>
const DALIMeta &TensorVector<Backend>::GetMeta(int idx) const {
  assert(static_cast<size_t>(idx) < curr_tensors_size_);
  return tensors_[idx]->GetMeta();
}


template <typename Backend>
void TensorVector<Backend>::SetMeta(int idx, const DALIMeta &meta) {
  assert(static_cast<size_t>(idx) < curr_tensors_size_);
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
    curr_tensors_size_ = 0;
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
  for (size_t i = 0; i < curr_tensors_size_; i++) {
    tensors_[i]->reserve(bytes_per_sample);
  }
}


template <typename Backend>
bool TensorVector<Backend>::IsContiguous() const noexcept {
  return state_ == State::contiguous && static_cast<size_t>(views_count_) == num_samples();
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
  curr_tensors_size_ = 0;
  type_ = {};
  if (IsContiguous()) {
    views_count_ = 0;
    tl_->Reset();
  }
}


template <typename Backend>
template <typename SrcBackend>
void TensorVector<Backend>::Copy(const TensorList<SrcBackend> &in_tl, AccessOrder order) {
  SetContiguous(true);
  type_ = in_tl.type_info();
  tl_->Copy(in_tl, order);

  resize_tensors(tl_->num_samples());
  UpdateViews();
}


template <typename Backend>
template <typename SrcBackend>
void TensorVector<Backend>::Copy(const TensorVector<SrcBackend> &in_tv, AccessOrder order) {
  SetContiguous(true);
  type_ = in_tv.type_;
  tl_->Copy(in_tv, order);

  resize_tensors(tl_->num_samples());
  UpdateViews();
}


template <typename Backend>
void TensorVector<Backend>::ShareData(const TensorList<Backend> &in_tl) {
  SetContiguous(true);
  type_ = in_tl.type_info();
  pinned_ = in_tl.is_pinned();
  tl_->ShareData(in_tl);

  resize_tensors(in_tl.num_samples());
  UpdateViews();
}

template <typename Backend>
void TensorVector<Backend>::ShareData(const TensorVector<Backend> &tv) {
  type_ = tv.type_;
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
    curr_tensors_size_ = other.curr_tensors_size_;
    tl_ = std::move(other.tl_);
    type_ = other.type_;
    views_count_ = other.views_count_.load();
    tensors_ = std::move(other.tensors_);
    for (auto &t : tensors_) {
      if (t) {
        if (auto *del = std::get_deleter<ViewRefDeleter>(t->data_)) del->ref = &views_count_;
      }
    }

    other.views_count_ = 0;
    other.curr_tensors_size_ = 0;
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

  assert(curr_tensors_size_ == tl_->num_samples());

  views_count_ = curr_tensors_size_;
  for (size_t i = 0; i < curr_tensors_size_; i++) {
    update_view(i);
  }
}


template <typename Backend>
std::shared_ptr<TensorList<Backend>> TensorVector<Backend>::AsTensorList(bool check_contiguity) {
  DALI_ENFORCE(IsContiguous() || !check_contiguity,
               "Cannot cast non continuous TensorVector to TensorList.");
  // Update the metadata when we are exposing the TensorList to the outside, as it might have been
  // kept in the individual tensors
  for (size_t idx = 0; idx < curr_tensors_size_; idx++) {
    tl_->SetMeta(idx, tensors_[idx]->GetMeta());
  }
  return tl_;
}


template <typename Backend>
void TensorVector<Backend>::resize_tensors(int new_size) {
  if (static_cast<size_t>(new_size) > tensors_.size()) {
    auto old_size = curr_tensors_size_;
    tensors_.resize(new_size);
    for (int i = old_size; i < new_size; i++) {
      if (!tensors_[i]) {
        tensors_[i] = std::make_shared<Tensor<Backend>>();
        tensors_[i]->set_pinned(is_pinned());
        tensors_[i]->set_order(order());
      }
    }
  } else if (static_cast<size_t>(new_size) < curr_tensors_size_) {
    for (size_t i = new_size; i < curr_tensors_size_; i++) {
      if (tensors_[i]->shares_data()) {
        tensors_[i]->Reset();
      }
    }
    // TODO(klecki): Do not keep the invalidated tensors - this prevents memory hogging but
    // also gets rid of reserved memory.
    // tensors_.resize(new_size);
  }
  curr_tensors_size_ = new_size;
}

template <typename Backend>
void TensorVector<Backend>::PropagateUp(bool contiguous) {
  // TODO(klecki): This is mostly simple consistency check, but most of the metadata will be moved
  // to the batch object for consitency and easier use in checks. It should allow for shape()
  // to be ready to use as well as easy verification for SetSample/CopySample.
  SetContiguous(contiguous);
  // assume that the curr_tensors_size_ is valid
  DALI_ENFORCE(curr_tensors_size_ > 0, "Unexpected empty output of operator. Internal DALI error.");
  type_ = tensors_[0]->type_info();
  pinned_ = tensors_[0]->is_pinned();
  order_ = tensors_[0]->order();
  tl_->set_order(order_);
  for (size_t i = 0; i < curr_tensors_size_; i++) {
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
void TensorVector<Backend>::update_view(int idx) {
  assert(static_cast<size_t>(idx) < curr_tensors_size_);
  assert(static_cast<size_t>(idx) < tl_->num_samples());

  auto *ptr = tl_->raw_mutable_tensor(idx);

  TensorShape<> shape = tl_->tensor_shape(idx);

  tensors_[idx]->Reset();
  // TODO(klecki): deleter that reduces views_count or just noop sharing?
  // tensors_[i]->ShareData(tl_.get(), static_cast<int>(idx));
  if (tensors_[idx]->raw_data() != ptr || tensors_[idx]->shape() != shape) {
    tensors_[idx]->ShareData(std::shared_ptr<void>(ptr, ViewRefDeleter{&views_count_}),
                             volume(tl_->tensor_shape(idx)) * tl_->type_info().size(),
                             tl_->is_pinned(),
                             shape, tl_->type(),
                             order());
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
