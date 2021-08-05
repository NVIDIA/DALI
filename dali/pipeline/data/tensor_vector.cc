// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  type_ = tl_->type();
  state_ = State::contiguous;
  resize_tensors(tl_->ntensor());
  UpdateViews();
}


template <typename Backend>
TensorVector<Backend>::TensorVector(TensorVector<Backend> &&other) noexcept {
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


template <typename Backend>
size_t TensorVector<Backend>::nbytes() const noexcept {
  if (state_ == State::contiguous) {
    return tl_->nbytes();
  }
  // else
  size_t total_nbytes = 0;
  for (const auto &t : tensors_) {
    total_nbytes += t->nbytes();
  }
  return total_nbytes;
}


template <typename Backend>
size_t TensorVector<Backend>::capacity() const noexcept {
  if (state_ == State::contiguous) {
    return tl_->capacity();
  }
  // else
  size_t total_capacity = 0;
  for (const auto &t : tensors_) {
    total_capacity += t->capacity();
  }
  return total_capacity;
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
void TensorVector<Backend>::Resize(const TensorListShape<> &new_shape, const TypeInfo &new_type) {
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
  if (new_size == 0) {
    Reset();
    return;
  }
  auto sh = shape();
  TensorListShape<> new_sh(new_size, sh.sample_dim());
  TensorShape<> zero_sh(sh[sh.num_samples() - 1]);  // 0-volume shape with given dimensionality
  for (int i = 0; i < zero_sh.size(); i++) {
    zero_sh[i] = 0;
  }

  int i = 0;
  for (; i < std::min(sh.num_samples(), new_size); i++) {
    new_sh.set_tensor_shape(i, sh[i]);
  }
  for (; i < new_size; i++) {
    new_sh.set_tensor_shape(i, zero_sh);
  }
  assert(!new_sh.empty());
  Resize(new_sh);
}


template <typename Backend>
void TensorVector<Backend>::set_type(const TypeInfo &new_type) {
  type_ = new_type;
  tl_->set_type(new_type);
  for (auto t : tensors_) {
    t->set_type(new_type);
  }
  if (state_ == State::contiguous) {
    UpdateViews();
  }
}


template <typename Backend>
const TypeInfo &TensorVector<Backend>::type() const {
  if (state_ == State::contiguous) {
    return tl_->type();
  }
  if (curr_tensors_size_ == 0) {
    return type_;
  }
  for (size_t i = 1; i < curr_tensors_size_; i++) {
    assert(tensors_[0]->type() == tensors_[i]->type());
  }
  return tensors_[0]->type();
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
  if (state_ == State::contiguous) {
    return tl_->is_pinned();
  }
  if (curr_tensors_size_ == 0) {
    return pinned_;
  }
  for (size_t i = 1; i < curr_tensors_size_; i++) {
    assert(tensors_[i]->is_pinned() == tensors_[0]->is_pinned());
  }
  return tensors_[0]->is_pinned();
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
  return state_ == State::contiguous && static_cast<size_t>(views_count_) == size();
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
void TensorVector<Backend>::Copy(const TensorList<SrcBackend> &in_tl, cudaStream_t stream) {
  SetContiguous(true);
  type_ = in_tl.type();
  tl_->Copy(in_tl, stream);

  resize_tensors(tl_->ntensor());
  UpdateViews();
}


template <typename Backend>
template <typename SrcBackend>
void TensorVector<Backend>::Copy(const TensorVector<SrcBackend> &in_tv, cudaStream_t stream) {
  SetContiguous(true);
  type_ = in_tv.type_;
  tl_->Copy(in_tv, stream);

  resize_tensors(tl_->ntensor());
  UpdateViews();
}


template <typename Backend>
void TensorVector<Backend>::ShareData(TensorList<Backend> *in_tl) {
  DALI_ENFORCE(in_tl != nullptr, "Input TensorList is nullptr");
  SetContiguous(true);
  type_ = in_tl->type();
  pinned_ = in_tl->is_pinned();
  tl_->ShareData(in_tl);

  resize_tensors(in_tl->ntensor());
  UpdateViews();
}


template <typename Backend>
void TensorVector<Backend>::ShareWith(TensorList<Backend> *in_tl) const {
  DALI_ENFORCE(in_tl != nullptr, "Input TensorList is nullptr");
  if (IsContiguous()) {
    in_tl->ShareData(tl_.get());
    for (size_t i = 0; i < size(); ++i) {
      in_tl->SetMeta(i, this->GetMeta(i));
    }
    in_tl->SetLayout(this->GetLayout());
  } else {
    DALI_FAIL("Cannot share non contiguous TensorVector with TensorList");
  }
}


template <typename Backend>
void TensorVector<Backend>::ShareData(TensorVector<Backend> *tv) {
  DALI_ENFORCE(tv != nullptr, "Input TensorVector is nullptr");
  type_ = tv->type_;
  state_ = tv->state_;
  pinned_ = tv->is_pinned();

  if (IsValidType(tv->tl_->type())) {
    tl_->ShareData(tv->tl_.get());
  } else {
    tl_->Reset();
    tl_->ResizeLike(*tv->tl_);
  }
  int batch_size = tv->ntensor();
  resize_tensors(batch_size);
  views_count_ = 0;

  for (int i = 0; i < batch_size; i++) {
    if (static_cast<int>(tv->tl_->ntensor()) > i &&
        tv->tensors_[i]->raw_data() == tv->tl_->raw_tensor(i)) {
      update_view(i);
      ++views_count_;
    } else {
      tensors_[i]->ShareData(tv->tensors_[i].get());
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
  type_ = tl_->type();

  assert(curr_tensors_size_ == tl_->ntensor());

  views_count_ = curr_tensors_size_;
  for (size_t i = 0; i < curr_tensors_size_; i++) {
    update_view(i);
  }
}


template <typename Backend>
std::shared_ptr<TensorList<Backend>> TensorVector<Backend>::AsTensorList(bool check_contiguity) {
  DALI_ENFORCE(IsContiguous() || !check_contiguity,
               "Cannot cast non continuous TensorVector to TensorList.");
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
      }
    }
  } else if (static_cast<size_t>(new_size) < curr_tensors_size_) {
    for (size_t i = new_size; i < curr_tensors_size_; i++) {
      if (tensors_[i]->shares_data()) {
        tensors_[i]->Reset();
      }
    }
  }
  curr_tensors_size_ = new_size;
}


template <typename Backend>
void TensorVector<Backend>::update_view(int idx) {
  assert(static_cast<size_t>(idx) < curr_tensors_size_);
  assert(static_cast<size_t>(idx) < tl_->ntensor());

  auto *ptr = tl_->raw_mutable_tensor(idx);

  TensorShape<> shape = tl_->tensor_shape(idx);

  tensors_[idx]->Reset();
  // TODO(klecki): deleter that reduces views_count or just noop sharing?
  // tensors_[i]->ShareData(tl_.get(), static_cast<int>(idx));
  if (tensors_[idx]->raw_data() != ptr || tensors_[idx]->shape() != shape) {
    tensors_[idx]->ShareData(std::shared_ptr<void>(ptr, ViewRefDeleter{&views_count_}),
                             volume(tl_->tensor_shape(idx)) * tl_->type().size(), shape,
                             tl_->type());
  } else if (IsValidType(tl_->type())) {
    tensors_[idx]->set_type(tl_->type());
  }
  tensors_[idx]->SetMeta(tl_->GetMeta(idx));
}


template class DLL_PUBLIC TensorVector<CPUBackend>;
template class DLL_PUBLIC TensorVector<GPUBackend>;
template void TensorVector<CPUBackend>::Copy<CPUBackend>(const TensorVector<CPUBackend>&, cudaStream_t);  // NOLINT
template void TensorVector<CPUBackend>::Copy<GPUBackend>(const TensorVector<GPUBackend>&, cudaStream_t);  // NOLINT
template void TensorVector<GPUBackend>::Copy<CPUBackend>(const TensorVector<CPUBackend>&, cudaStream_t);  // NOLINT
template void TensorVector<GPUBackend>::Copy<GPUBackend>(const TensorVector<GPUBackend>&, cudaStream_t);  // NOLINT
template void TensorVector<CPUBackend>::Copy<CPUBackend>(const TensorList<CPUBackend>&, cudaStream_t);  // NOLINT
template void TensorVector<CPUBackend>::Copy<GPUBackend>(const TensorList<GPUBackend>&, cudaStream_t);  // NOLINT
template void TensorVector<GPUBackend>::Copy<CPUBackend>(const TensorList<CPUBackend>&, cudaStream_t);  // NOLINT
template void TensorVector<GPUBackend>::Copy<GPUBackend>(const TensorList<GPUBackend>&, cudaStream_t);  // NOLINT

}  // namespace dali
