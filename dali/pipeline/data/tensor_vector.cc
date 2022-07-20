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

#include "dali/core/access_order.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/tensor_layout.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/tensor_vector.h"
#include "dali/pipeline/data/types.h"

namespace dali {

namespace {

/**
 * @brief Check if both shared pointers have the same managed pointer (not the one returned by
 * .get())
 */
bool same_owner(const std::shared_ptr<void> &x, const std::shared_ptr<void> &y) {
  if (x.owner_before(y) || y.owner_before(x))
    return false;
  return true;
}


bool same_owner(const std::weak_ptr<void> &x, const std::shared_ptr<void> &y) {
  if (x.owner_before(y) || y.owner_before(x))
    return false;
  return true;
}


// TODO(klecki): move this to the class?
TensorShape<> empty_shape(int dim) {
  TensorShape<> result;
  result.resize(dim);
  for (auto &elem : result) {
    elem = 0;
  }
  return result;
}

}  // namespace

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
TensorVector<Backend>::TensorVector() : curr_num_tensors_(0) {}


template <typename Backend>
TensorVector<Backend>::TensorVector(int batch_size) : curr_num_tensors_(0) {
  // This is why we can't have nice things as `dim = 0` is already occupied
  // by competing functionality, we need to guard ourself from thinking we have some scalar
  // allocation where in fact we just have empty samples.
  // So instead we set the dim to 1, and the resize_tensor will cause the shape to be resized
  // to appropriate number of samples and 0-initialized, so copy from batch to batch,
  // using this as (empty) source still works. Maybe there is a better solution to this problem.
  set_sample_dim(1);
  resize_tensors(batch_size);
}


template <typename Backend>
TensorVector<Backend>::TensorVector(TensorVector<Backend> &&other) noexcept {
  *this = std::move(other);
}


template <typename Backend>
TensorVector<Backend> &TensorVector<Backend>::operator=(TensorVector<Backend> &&other) noexcept {
  if (&other != this) {
    contiguous_buffer_ = std::move(other.contiguous_buffer_);
    buffer_bkp_ = std::move(other.buffer_bkp_);
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


// This is to check if we are actually laid down in contiguous memory
// TODO(klecki): make this internal and name it something like: IsContiguouslyStored?
template <typename Backend>
bool TensorVector<Backend>::IsContiguousTensor() const {
  if (num_samples() == 0 || shape().num_elements() == 0) {
    return true;
  }
  // If we are using contiguous representation in this case we can safely return
  if (IsContiguous()) {
    return true;
  }
  const uint8_t *base_ptr = static_cast<const uint8_t *>(tensors_[0].raw_data());
  size_t size = type_info().size();

  for (int i = 0; i < shape_.size(); ++i) {
    if (base_ptr != tensors_[i].raw_data()) {
      return false;
    }
    base_ptr += shape_[i].num_elements() * size;
  }
  return true;
}


template <typename Backend>
bool TensorVector<Backend>::IsDenseTensor() const {
  return IsContiguous() && is_uniform(shape());
}


template <typename Backend>
Tensor<Backend> TensorVector<Backend>::AsReshapedTensor(const TensorShape<> &new_shape) {
  DALI_ENFORCE(num_samples() > 0,
               "To create a view Tensor, the batch must have at least 1 element.");
  DALI_ENFORCE(IsValidType(type()),
               "To create a view Tensor, the batch must have a valid data type.");
  DALI_ENFORCE(
      shape().num_elements() == new_shape.num_elements(),
      make_string("To create a view Tensor, requested shape need to have the same volume as the "
                  "batch, requested: ",
                  new_shape.num_elements(), " expected: ", shape().num_elements()));
  Tensor<Backend> result;
  result.ShareData(contiguous_buffer_.get_data_ptr(), contiguous_buffer_.capacity(),
                   contiguous_buffer_.is_pinned(), new_shape, type(), device_id(), order());
  auto result_layout = GetLayout();
  if (!GetLayout().empty()) {
    result_layout = TensorLayout("N") + result_layout;
  }
  result.SetLayout(result_layout);
  return result;
}


template <typename Backend>
Tensor<Backend> TensorVector<Backend>::AsTensor() {
  DALI_ENFORCE(IsDenseTensor(),
               "The batch must be representable tensor - it must has uniform shape and be "
               "allocated in contiguous memory.");
  DALI_ENFORCE(shape().num_samples() > 0,
               "To create a view Tensor, the batch must have at least 1 element.");
  return AsReshapedTensor(shape_cat(shape().num_samples(), shape()[0]));
}


template <typename Backend>
void TensorVector<Backend>::VerifySampleShareConformance(DALIDataType type, int sample_dim,
                                                         TensorLayout layout, bool pinned,
                                                         AccessOrder order, int device_id,
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

  DALI_ENFORCE(this->order() == order,
               make_string("Sample must have the same order as the target batch", error_suffix));

  DALI_ENFORCE(this->device_id() == device_id,
               make_string("Sample must have the same device id as target batch, current: ",
                           this->device_id(), ", new: ", device_id, error_suffix));
}


template <typename Backend>
void TensorVector<Backend>::UnsafeSetSample(int sample_idx, const TensorVector<Backend> &src,
                                            int src_sample_idx) {
  // Bounds check
  assert(sample_idx >= 0 && sample_idx < curr_num_tensors_);
  assert(src_sample_idx >= 0 && src_sample_idx < src.curr_num_tensors_);
  // Setting any individual sample converts the batch to non-contiguous mode
  MakeNoncontiguous();
  if (&src.tensors_[src_sample_idx] == &tensors_[sample_idx])
    return;
  VerifySampleShareConformance(src.type(), src.shape().sample_dim(), src.GetLayout(),
                               src.is_pinned(), src.order(), src.device_id(),
                               make_string(" for ", sample_idx, " <- ", src_sample_idx, "."));

  shape_.set_tensor_shape(sample_idx, src.shape()[src_sample_idx]);

  // Setting a new share overwrites the previous one - so we can safely assume that even if
  // we had a sample sharing into TL, it will be overwritten
  tensors_[sample_idx].ShareData(src.tensors_[src_sample_idx]);

  if (src.GetLayout().empty() && !GetLayout().empty()) {
    tensors_[sample_idx].SetLayout(GetLayout());
  }
}


template <typename Backend>
void TensorVector<Backend>::UnsafeSetSample(int sample_idx, const Tensor<Backend> &owner) {
  // Bounds check
  assert(sample_idx >= 0 && sample_idx < curr_num_tensors_);
  // Setting any individual sample converts the batch to non-contiguous mode
  MakeNoncontiguous();
  VerifySampleShareConformance(owner.type(), owner.shape().sample_dim(), owner.GetLayout(),
                               owner.is_pinned(), owner.order(), owner.device_id(),
                               make_string(" for ", sample_idx, "."));

  shape_.set_tensor_shape(sample_idx, owner.shape());

  // Setting a new share overwrites the previous one - so we can safely assume that even if
  // we had a sample sharing into TL, it will be overwritten
  tensors_[sample_idx].ShareData(owner);

  if (owner.GetLayout().empty() && !GetLayout().empty()) {
    tensors_[sample_idx].SetLayout(GetLayout());
  }
}


template <typename Backend>
void TensorVector<Backend>::UnsafeSetSample(int sample_idx, const shared_ptr<void> &ptr,
                                            size_t bytes, bool pinned, const TensorShape<> &shape,
                                            DALIDataType type, int device_id, AccessOrder order,
                                            const TensorLayout &layout) {
  // Bounds check
  assert(sample_idx >= 0 && sample_idx < curr_num_tensors_);
  // Setting any individual sample converts the batch to non-contiguous mode
  MakeNoncontiguous();
  VerifySampleShareConformance(type, shape.sample_dim(), layout, pinned, order, device_id,
                               make_string(" for ", sample_idx, "."));

  DALI_ENFORCE(!IsContiguous());
  shape_.set_tensor_shape(sample_idx, shape);

  // Setting a new share overwrites the previous one - so we can safely assume that even if
  // we had a sample sharing into TL, it will be overwritten
  tensors_[sample_idx].ShareData(ptr, bytes, pinned, shape, type, device_id, order);

  if (layout.empty() && !GetLayout().empty()) {
    tensors_[sample_idx].SetLayout(GetLayout());
  }
}


template <typename Backend>
void TensorVector<Backend>::VerifySampleCopyConformance(DALIDataType type, int sample_dim,
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
void TensorVector<Backend>::UnsafeCopySample(int sample_idx, const TensorVector<Backend> &src,
                                             int src_sample_idx, AccessOrder order) {
  // Bounds check
  assert(sample_idx >= 0 && sample_idx < curr_num_tensors_);
  assert(src_sample_idx >= 0 && src_sample_idx < src.curr_num_tensors_);
  VerifySampleCopyConformance(src.type(), src.shape().sample_dim(), src.GetLayout(),
                              shape()[sample_idx], src.shape()[src_sample_idx],
                              make_string(" for ", sample_idx, " <- ", src_sample_idx, "."));

  shape_.set_tensor_shape(sample_idx, src.shape()[src_sample_idx]);
  tensors_[sample_idx].Copy(src.tensors_[src_sample_idx], order);
  if (src.GetLayout().empty() && !GetLayout().empty()) {
    tensors_[sample_idx].SetLayout(GetLayout());
  }
}


template <typename Backend>
void TensorVector<Backend>::UnsafeCopySample(int sample_idx, const Tensor<Backend> &src,
                                             AccessOrder order) {
  // Bounds check
  assert(sample_idx >= 0 && sample_idx < curr_num_tensors_);
  VerifySampleCopyConformance(src.type(), src.shape().sample_dim(), src.GetLayout(),
                              shape()[sample_idx], src.shape(),
                              make_string(" for ", sample_idx, "."));

  shape_.set_tensor_shape(sample_idx, src.shape());
  tensors_[sample_idx].Copy(src, order);
  if (src.GetLayout().empty() && !GetLayout().empty()) {
    tensors_[sample_idx].SetLayout(GetLayout());
  }
}


template <typename Backend>
void TensorVector<Backend>::set_sample_dim(int sample_dim) {
  DALI_ENFORCE(
      !has_data(),
      "Setting sample dim is not allowed when batch is already allocated, use Resize instead.");
  sample_dim_ = sample_dim;
  shape_.resize(shape_.num_samples(), sample_dim);
}


template <typename Backend>
size_t TensorVector<Backend>::nbytes() const noexcept {
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
size_t TensorVector<Backend>::capacity() const noexcept {
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
std::vector<size_t> TensorVector<Backend>::_chunks_nbytes() const {
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
std::vector<size_t> TensorVector<Backend>::_chunks_capacity() const {
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
const TensorListShape<> &TensorVector<Backend>::shape() const & {
  return shape_;
}


template <typename Backend>
void TensorVector<Backend>::set_order(AccessOrder order, bool synchronize) {
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
      this->order().wait(order);
  }
  contiguous_buffer_.set_order(order, false);
  for (auto &t : tensors_)
    t.set_order(order, false);
  order_ = order;
}


template <typename Backend>
SampleView<Backend> TensorVector<Backend>::operator[](size_t pos) {
  DALI_ENFORCE(pos < static_cast<size_t>(curr_num_tensors_), "Out of bounds access");
  return {tensors_[pos].raw_mutable_data(), shape().tensor_shape_span(pos), tensors_[pos].type()};
}


template <typename Backend>
ConstSampleView<Backend> TensorVector<Backend>::operator[](size_t pos) const {
  DALI_ENFORCE(pos < static_cast<size_t>(curr_num_tensors_), "Out of bounds access");
  return {tensors_[pos].raw_data(), shape().tensor_shape_span(pos), tensors_[pos].type()};
}


template <typename Backend>
void TensorVector<Backend>::Resize(const TensorListShape<> &new_shape, DALIDataType new_type,
                                   BatchState state) {
  DALI_ENFORCE(IsValidType(new_type),
               "TensorVector cannot be resized with invalid type. To zero out the TensorVector "
               "Reset() can be used.");
  if (state_.Update(state)) {
    if (!state_.IsContiguous()) {
      // As we updated the state to noncontiguous, we need to detach the buffers
      DoMakeNoncontiguous();
    }
  }
  resize_tensors(new_shape.num_samples());
  if (type_.id() != new_type) {
    type_ = TypeTable::GetTypeInfo(new_type);
    // calling appropriate resize and/or recreate_views propagates type to individual samples
  }
  sample_dim_ = new_shape.sample_dim();
  shape_ = new_shape;

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
DALIDataType TensorVector<Backend>::type() const {
  return type_.id();
}


template <typename Backend>
const TypeInfo &TensorVector<Backend>::type_info() const {
  return type_;
}


template <typename Backend>
void TensorVector<Backend>::SetLayout(const TensorLayout &layout) {
  for (auto &t : tensors_) {
    t.SetLayout(layout);
  }
  layout_ = layout;
}


template <typename Backend>
void TensorVector<Backend>::SetSkipSample(int idx, bool skip_sample) {
  tensors_[idx].SetSkipSample(skip_sample);
}


template <typename Backend>
void TensorVector<Backend>::SetSourceInfo(int idx, const std::string &source_info) {
  tensors_[idx].SetSourceInfo(source_info);
}


template <typename Backend>
TensorLayout TensorVector<Backend>::GetLayout() const {
  return layout_;
}


template <typename Backend>
const DALIMeta &TensorVector<Backend>::GetMeta(int idx) const {
  assert(idx < curr_num_tensors_);
  return tensors_[idx].GetMeta();
}


template <typename Backend>
void TensorVector<Backend>::SetMeta(int idx, const DALIMeta &meta) {
  assert(idx < curr_num_tensors_);
  tensors_[idx].SetMeta(meta);
}


template <typename Backend>
void TensorVector<Backend>::set_pinned(bool pinned) {
  contiguous_buffer_.set_pinned(pinned);
  for (auto &t : tensors_) {
    t.set_pinned(pinned);
  }
  pinned_ = pinned;
}


template <typename Backend>
bool TensorVector<Backend>::is_pinned() const {
  return pinned_;
}


template <typename Backend>
void TensorVector<Backend>::set_device_id(int device_id) {
  contiguous_buffer_.set_device_id(device_id);
  for (auto &t : tensors_) {
    t.set_device_id(device_id);
  }
  device_ = device_id;
}


template <typename Backend>
int TensorVector<Backend>::device_id() const {
  return device_;
}


template <typename Backend>
void TensorVector<Backend>::reserve(size_t total_bytes) {
  int batch_size_bkp = curr_num_tensors_;
  if (!state_.IsContiguous()) {
    tensors_.clear();
    resize_tensors(0);
  }
  state_.Setup(BatchState::Contiguous);
  contiguous_buffer_.reserve(total_bytes);
  if (IsValidType(type_)) {
    resize_tensors(batch_size_bkp);
    recreate_views();
  }
}


template <typename Backend>
void TensorVector<Backend>::reserve(size_t bytes_per_sample, int batch_size) {
  assert(batch_size > 0);
  state_.Setup(BatchState::Noncontiguous);
  resize_tensors(batch_size);
  for (int i = 0; i < curr_num_tensors_; i++) {
    tensors_[i].reserve(bytes_per_sample);
  }
}


template <typename Backend>
bool TensorVector<Backend>::IsContiguous() const noexcept {
  return state_.IsContiguous();
}


template <typename Backend>
void TensorVector<Backend>::recreate_views() {
  // precondition: type, shape are configured
  uint8_t *base_ptr = static_cast<uint8_t *>(contiguous_buffer_.raw_mutable_data());
  int64_t num_samples = shape().num_samples();
  for (int64_t i = 0; i < num_samples; i++) {
    // or any other way
    auto tensor_size = shape().tensor_size(i);

    std::shared_ptr<void> sample_alias(contiguous_buffer_.get_data_ptr(), base_ptr);
    tensors_[i].ShareData(sample_alias, tensor_size * type_info().size(), is_pinned(), shape()[i],
                          type(), device_id(), order());
    tensors_[i].set_device_id(device_id());
    base_ptr += tensor_size * type_info().size();
  }
}


template <typename Backend>
void TensorVector<Backend>::SetContiguous(BatchState state) {
  if (state == BatchState::Default) {
    // remove the force, keep the current state information
    state_.Setup(state_.Get(), false);
    return;
  }
  DALI_ENFORCE(state_.Get() == state || !has_data(),
               "Contiguous or non-contiguous mode cannot be set to already allocated buffer.");
  state_.Setup(state, true);
}


template <typename Backend>
void TensorVector<Backend>::SetContiguous(bool state) {
  SetContiguous(state ? BatchState::Contiguous : BatchState::Noncontiguous);
}


template <typename Backend>
void TensorVector<Backend>::MakeContiguous(std::weak_ptr<void> owner) {
  if (state_.IsContiguous()) {
    return;
  }
  DALI_FAIL("Don't know how to coalesce the buffer yet");
}


template <typename Backend>
void TensorVector<Backend>::MakeNoncontiguous() {
  if (!state_.IsContiguous()) {
    return;
  }

  state_.Update(BatchState::Noncontiguous);
  DoMakeNoncontiguous();
}


template <typename Backend>
void TensorVector<Backend>::DoMakeNoncontiguous() {
  // We clear the contiguous_buffer_, as we are now non-contiguous.
  buffer_bkp_ = contiguous_buffer_.get_data_ptr();
  contiguous_buffer_.reset();
  for (auto &t : tensors_) {
    // If the Tensor was aliasing the contiguous buffer, mark it as not sharing any data.
    // This will allow for the individual buffers to be resized.
    // The downside of this is we may keep the big contiguous buffer until all individual
    // samples are replaced.
    if (same_owner(buffer_bkp_, t.data_)) {
      t.detach();
    }
  }
}


template <typename Backend>
void TensorVector<Backend>::Reset() {
  if (IsContiguous()) {
    contiguous_buffer_.reset();
  }
  buffer_bkp_.reset();
  // TODO(klecki): Is there any benefit to call Reset on all?
  tensors_.clear();

  curr_num_tensors_ = 0;
  type_ = {};
  sample_dim_ = -1;
  shape_ = {};
  layout_ = "";
  // N.B. state_, pinned_, order_ and device_ are not reset here, as they might be previously set
  // up via the executor - TODO(klecki) - consider if we want to keep this behaviour
}


template <typename Backend>
template <typename SrcBackend>
void TensorVector<Backend>::Copy(const TensorList<SrcBackend> &src, AccessOrder order) {
  // This variant will be removed with the removal of TensorList.

  auto copy_order = copy_impl::SyncBefore(this->order(), src.order(), order);

  Resize(src.shape(), src.type());
  // After resize the state_, curr_num_tensors_, type_, sample_dim_, shape_ (and pinned)
  // postconditions are met, as well as the buffers are correctly adjusted.

  copy_impl::SyncAfterResize(this->order(), copy_order);

  bool use_copy_kernel = false;
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
template <typename SrcBackend>
void TensorVector<Backend>::Copy(const TensorVector<SrcBackend> &src, AccessOrder order,
                                 bool use_copy_kernel) {
  auto copy_order = copy_impl::SyncBefore(this->order(), src.order(), order);

  Resize(src.shape(), src.type());
  // After resize the state_, curr_num_tensors_, type_, sample_dim_, shape_ (and pinned)
  // postconditions are met, as well as the buffers are correctly adjusted.

  copy_impl::SyncAfterResize(this->order(), copy_order);

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
void TensorVector<Backend>::ShareData(const TensorList<Backend> &in_tl) {
  Reset();

  state_.Setup(BatchState::Contiguous);
  curr_num_tensors_ = in_tl.num_samples();
  type_ = in_tl.type_info();
  sample_dim_ = in_tl.shape().sample_dim();
  shape_ = in_tl.shape();
  layout_ = in_tl.GetLayout();
  pinned_ = in_tl.is_pinned();
  order_ = in_tl.order();
  device_ = in_tl.device_id();

  contiguous_buffer_.ShareData(in_tl.data_);
  // Create empty tensors by hand so we do not allocate twice as the shape is already set
  tensors_.resize(in_tl.num_samples());
  // recrete the aliases
  recreate_views();

  SetLayout(in_tl.GetLayout());
  for (int i = 0; i < curr_num_tensors_; i++) {
    SetMeta(i, in_tl.GetMeta(i));
  }
}


template <typename Backend>
void TensorVector<Backend>::ShareData(const TensorVector<Backend> &tv) {
  Reset();

  state_ = tv.state_;
  curr_num_tensors_ = tv.curr_num_tensors_;
  type_ = tv.type_;
  sample_dim_ = tv.sample_dim_;
  shape_ = tv.shape_;
  layout_ = tv.layout_;
  pinned_ = tv.pinned_;
  order_ = tv.order_;
  device_ = tv.device_;

  if (tv.IsContiguous()) {
    contiguous_buffer_.ShareData(tv.contiguous_buffer_);
    tensors_.resize(shape().num_samples());
    recreate_views();
  } else {
    int batch_size = tv.num_samples();
    tensors_.resize(shape().num_samples());
    for (int i = 0; i < batch_size; i++) {
      tensors_[i].ShareData(tv.tensors_[i]);
    }
  }

  SetLayout(tv.GetLayout());
  for (int i = 0; i < curr_num_tensors_; i++) {
    SetMeta(i, tv.GetMeta(i));
  }
}


template <typename Backend>
void TensorVector<Backend>::ShareData(const shared_ptr<void> &ptr, size_t bytes, bool pinned,
                                      const TensorListShape<> &shape, DALIDataType type,
                                      int device_id, AccessOrder order,
                                      const TensorLayout &layout) {
  contiguous_buffer_.set_backing_allocation(ptr, bytes, pinned, type, shape.num_elements(),
                                            device_id, order);
  buffer_bkp_.reset();
  tensors_.clear();
  tensors_.resize(shape.num_samples());

  state_.Update(BatchState::Contiguous);
  curr_num_tensors_ = shape.num_samples();
  type_ = TypeTable::GetTypeInfo(type);
  sample_dim_ = shape.sample_dim();
  shape_ = shape;
  layout_ = layout;
  pinned_ = pinned;
  device_ = device_id;
  order_ = order;
  recreate_views();
}


template <typename Backend>
void TensorVector<Backend>::resize_tensors(int new_size) {
  // This doesn't update with the same order as the class members are listed
  // We need to make sure everything is updated for the tensors that come back into scope
  // and we start with the pinned and order properties as they might impact future allocations.
  // next we make sure the type is consistent, and if so, introduce empty shape
  shape_.resize(new_size);
  if (new_size > curr_num_tensors_) {
    auto old_size = curr_num_tensors_;
    tensors_.resize(new_size);
    for (int i = old_size; i < new_size; i++) {
      // TODO(klecki) same validation as when updating properties - or reset
      if (!tensors_[i].has_data()) {
        tensors_[i].set_pinned(is_pinned());
      } else {
        DALI_ENFORCE(tensors_[i].is_pinned() == is_pinned());
      }
      tensors_[i].set_order(order());
      tensors_[i].set_device_id(device_id());
      if (type() != DALI_NO_TYPE) {
        tensors_[i].set_type(type());
        if (sample_dim_ >= 0) {
          tensors_[i].Resize(empty_shape(sample_dim()));
          shape_.set_tensor_shape(i, empty_shape(sample_dim()));
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
void TensorVector<Backend>::UpdatePropertiesFromSamples(bool contiguous) {
  state_.Update(contiguous ? BatchState::Contiguous : BatchState::Noncontiguous);
  // assume that the curr_num_tensors_ is valid
  DALI_ENFORCE(curr_num_tensors_ > 0,
               "Unexpected empty output of per-sample operator. Internal DALI error.");
  type_ = tensors_[0].type_info();
  sample_dim_ = tensors_[0].shape().sample_dim();
  shape_.resize(curr_num_tensors_, sample_dim_);
  layout_ = tensors_[0].GetMeta().GetLayout();
  pinned_ = tensors_[0].is_pinned();
  order_ = tensors_[0].order();
  device_ = tensors_[0].device_id();
  contiguous_buffer_.set_order(order_);
  for (int i = 0; i < curr_num_tensors_; i++) {
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
bool TensorVector<Backend>::has_data() const {
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
bool TensorVector<Backend>::shares_data() const {
  // TODO(klecki): I would like to get rid of some of this
  if (IsContiguous()) {
    return contiguous_buffer_.shares_data();
  }
  for (const auto &tensor : tensors_) {
    if (tensor.shares_data() && !same_owner(contiguous_buffer_.get_data_ptr(),
                                            tensor.get_data_ptr())) {
      return true;
    }
  }
  return false;
}


template class DLL_PUBLIC TensorVector<CPUBackend>;
template class DLL_PUBLIC TensorVector<GPUBackend>;
template void TensorVector<CPUBackend>::Copy<CPUBackend>(const TensorVector<CPUBackend>&, AccessOrder, bool);  // NOLINT
template void TensorVector<CPUBackend>::Copy<GPUBackend>(const TensorVector<GPUBackend>&, AccessOrder, bool);  // NOLINT
template void TensorVector<GPUBackend>::Copy<CPUBackend>(const TensorVector<CPUBackend>&, AccessOrder, bool);  // NOLINT
template void TensorVector<GPUBackend>::Copy<GPUBackend>(const TensorVector<GPUBackend>&, AccessOrder, bool);  // NOLINT
template void TensorVector<CPUBackend>::Copy<CPUBackend>(const TensorList<CPUBackend>&, AccessOrder);  // NOLINT
template void TensorVector<CPUBackend>::Copy<GPUBackend>(const TensorList<GPUBackend>&, AccessOrder);  // NOLINT
template void TensorVector<GPUBackend>::Copy<CPUBackend>(const TensorList<CPUBackend>&, AccessOrder);  // NOLINT
template void TensorVector<GPUBackend>::Copy<GPUBackend>(const TensorList<GPUBackend>&, AccessOrder);  // NOLINT

}  // namespace dali
