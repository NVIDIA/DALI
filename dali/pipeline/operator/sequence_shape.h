// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_SEQUENCE_SHAPE_H_
#define DALI_PIPELINE_OPERATOR_SEQUENCE_SHAPE_H_

#include <memory>
#include <string>
#include <utility>
#include "dali/kernels/common/scatter_gather.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/data/sequence_utils.h"

namespace dali {

/**
 * @brief Describes expandable prefix of the layout and of the shape of a batch.
 * For instance, for batch of FHWC samples, expandable prefix is "F", i.e. each sample
 * should be expanded into number of HWC samples. If ``should_expand_channels`` is true,
 * the expandable prefix can consits of two extents, for example for FCHW input,
 * the "FC" extents are expandable.
 */
class ExpandDesc {
 public:
  ExpandDesc(const TensorListShape<> &shape, TensorLayout layout, bool should_expand_channels)
      : layout_{std::move(layout)},
        frames_dim_{VideoLayoutInfo::FrameDimIndex(layout_)},
        channels_dim_{VideoLayoutInfo::ChannelDimIndex(layout_)},
        expand_frames_{frames_dim_ == 0 ||
                       (should_expand_channels && frames_dim_ == 1 && channels_dim_ == 0)},
        expand_channels_{should_expand_channels && 0 <= channels_dim_ && channels_dim_ <= 1},
        num_expand_dims_{expand_channels_ + expand_frames_},
        is_channel_first_{num_expand_dims_ == 2 && channels_dim_ < frames_dim_},
        dims_to_expand_{shape.first(num_expand_dims_)},
        num_expanded_{dims_to_expand_.num_elements()} {}

  bool ExpandChannels() const {
    return expand_channels_;
  }

  bool ExpandFrames() const {
    return expand_frames_;
  }

  bool IsChannelFirst() const {
    return is_channel_first_;
  }

  int NumDimsToExpand() const {
    return num_expand_dims_;
  }

  int NumSamples() const {
    return dims_to_expand_.num_samples();
  }

  int64_t NumExpanded() const {
    return num_expanded_;
  }

  int64_t NumExpanded(int sample_idx) const {
    assert(sample_idx < NumSamples());
    return volume(dims_to_expand_[sample_idx]);
  }

  int64_t NumFrames(int sample_idx) const {
    assert(0 <= sample_idx && sample_idx < NumSamples());
    assert(frames_dim_ < NumDimsToExpand());
    return dims_to_expand_[sample_idx][frames_dim_];
  }

  int64_t NumChannels(int sample_idx) const {
    assert(0 <= sample_idx && sample_idx < NumSamples());
    assert(channels_dim_ < NumDimsToExpand());
    return dims_to_expand_[sample_idx][channels_dim_];
  }

  const TensorListShape<> &DimsToExpand() const {
    return dims_to_expand_;
  }

  TensorLayout Layout() const {
    return layout_;
  }

  TensorLayout ExpandedLayout() const {
    return layout_.first(num_expand_dims_);
  }

 private:
  TensorLayout layout_;
  int frames_dim_;
  int channels_dim_;
  bool expand_frames_;
  bool expand_channels_;
  int num_expand_dims_;
  bool is_channel_first_;
  TensorListShape<> dims_to_expand_;
  int64_t num_expanded_;
};

namespace sequence_utils {

struct SliceView {
  uint8_t *ptr;
  TensorShape<> shape;
  size_t type_size;
};


/**
 * @brief Iterable range over slices of tensor, created by expansion of the outermost
 * extents of the tensor. For instance passing ``SliceView`` of shape ``{6, 5, 3, 4}``
 * and ``ndims_to_unfold=2`` will result in range of 30 ``SliceViews`` of shape ``{3, 4}`` each.
 */
class UnfoldedSliceRange {
 public:
  using IndexType = int64_t;
  using ValueType = SliceView;

  UnfoldedSliceRange(ValueType view, int ndims_to_unfold)
      : view_{std::move(view)},
        num_slices_{[&]() {
          assert(view_.shape.sample_dim() >= ndims_to_unfold);
          return volume(view_.shape.begin(), view_.shape.begin() + ndims_to_unfold);
        }()},
        slice_shape_{view_.shape.begin() + ndims_to_unfold, view_.shape.end()},
        slice_stride_{view_.type_size * volume(slice_shape_)} {}

  RangeIterator<UnfoldedSliceRange> begin() const {
    return {*this, 0};
  }

  RangeIterator<UnfoldedSliceRange> end() const {
    return {*this, NumSlices()};
  }

  ValueType operator[](IndexType idx) const {
    return {view_.ptr + idx * SliceSize(), SliceShape(), view_.type_size};
  }

  int64_t NumSlices() const {
    return num_slices_;
  }

  TensorShape<> SliceShape() const {
    return slice_shape_;
  }

  size_t SliceSize() const {
    return slice_stride_;
  }

 private:
  ValueType view_;
  int64_t num_slices_;
  TensorShape<> slice_shape_;
  size_t slice_stride_;
};

template <typename BatchType>
TensorLayout unfolded_sample_layout(const BatchType &batch, int ndims_to_unfold) {
  assert(batch.sample_dim() >= ndims_to_unfold);
  auto sample_dim = batch.sample_dim() - ndims_to_unfold;
  const auto &initial_layout = batch.GetLayout();
  return initial_layout.empty() ? "" : initial_layout.last(sample_dim);
}

template <typename BatchType>
void setup_expanded_like(BatchType &expanded_batch, const BatchType &batch) {
  expanded_batch.set_pinned(batch.is_pinned());
  expanded_batch.set_order(batch.order());
}

template <typename BatchType>
std::shared_ptr<BatchType> expanded_like(const BatchType &batch) {
  auto expanded_handle = std::make_shared<BatchType>();
  setup_expanded_like(*expanded_handle, batch);
  return expanded_handle;
}

/**
 * @brief Utility to set shared samples in tensor vector from slices of samples of
 * another tensor vector.
 */
template <typename Backend>
struct TensorVectorBuilder {
  TensorVectorBuilder(TensorVector<Backend> &tv) : tv_{tv} {}  // NOLINT

  void SetNext(const SliceView &view) {
    assert(NextSampleIdx() < tv_.num_samples());
    std::shared_ptr<void> ptr(view.ptr, [](void *) {});  // no deleter
    tv_.UnsafeSetSample(next_++, ptr, view.type_size * volume(view.shape), tv_.is_pinned(),
                        view.shape, tv_.type(), tv_.order(), tv_.GetLayout());
  }

  int NextSampleIdx() const {
    return next_;
  }

 private:
  TensorVector<Backend> &tv_;
  int next_ = 0;
};

template <typename Backend>
TensorVectorBuilder<Backend> tv_builder_like(TensorVector<Backend> &expanded_batch,
                                             const TensorVector<Backend> &batch,
                                             int num_expanded_samples, int ndims_to_unfold = 0) {
  assert(batch.sample_dim() >= ndims_to_unfold);
  expanded_batch.Reset();
  expanded_batch.SetSize(num_expanded_samples);
  expanded_batch.set_sample_dim(batch.sample_dim() - ndims_to_unfold);
  expanded_batch.set_type(batch.type());
  expanded_batch.SetLayout(unfolded_sample_layout(batch, ndims_to_unfold));
  return {expanded_batch};
}

template <typename Backend>
UnfoldedSliceRange unfolded_slice_range(const TensorVector<Backend> &data, int sample_idx,
                                        int ndims_to_unfold) {
  const auto &type_info = data.type_info();
  auto type_size = type_info.size();
  const auto &shape = data.shape();
  uint8_t *base_ptr =
      const_cast<uint8_t *>(static_cast<const uint8_t *>(data.raw_tensor(sample_idx)));
  return {{base_ptr, shape[sample_idx], type_size}, ndims_to_unfold};
}

/** @defgroup broadcast_samples Utilities to broadcast `batch/shape` according to `expand_extents`.
 * The functions assume that the number of samples in the source `batch` is equal to number of samples
 * in the `expand_extents`. Then, the i-th sample of the source batch is repeated
 * `volume(expand_extents[i])` times in the destination batch. As a result, the destination batch
 * ends up with `expand_extents.num_elements()` samples.
 *
 * The TensorVector specialization is implemented in terms of pointers - the underlying data is not copied.
 * The TensorList specializations do copy data - due to the contiguity requirements.
 * @{
 */

inline TensorListShape<> broadcast_sample_shapes(const TensorListShape<> &shape,
                                                 int num_expanded_samples,
                                                 const TensorListShape<> &expand_extents) {
  assert(num_expanded_samples == expand_extents.num_elements());
  assert(expand_extents.num_samples() == shape.num_samples());
  TensorListShape<> broadcast(num_expanded_samples, shape.sample_dim());
  for (int sample_idx = 0, elem_idx = 0; sample_idx < shape.num_samples(); sample_idx++) {
    const auto &sample_shape = shape[sample_idx];
    for (int i = 0; i < volume(expand_extents[sample_idx]); i++) {
      broadcast.set_tensor_shape(elem_idx++, sample_shape);
    }
  }
  return broadcast;
}

template <typename Backend>
void broadcast_samples(TensorVector<Backend> &expanded_batch, const TensorVector<Backend> &batch,
                       int num_expanded_samples, const TensorListShape<> &expand_extents) {
  assert(num_expanded_samples == expand_extents.num_elements());
  assert(expand_extents.num_samples() == batch.num_samples());
  auto expanded_builder = tv_builder_like(expanded_batch, batch, num_expanded_samples);
  const auto &shape = batch.shape();
  const auto &type_info = batch.type_info();
  for (int sample_idx = 0; sample_idx < batch.num_samples(); sample_idx++) {
    const auto &slice_shape = shape[sample_idx];
    int num_elements = volume(expand_extents[sample_idx]);
    uint8_t *ptr =
        const_cast<uint8_t *>(static_cast<const uint8_t *>(batch.raw_tensor(sample_idx)));
    for (int sample_slice_idx = 0; sample_slice_idx < num_elements; sample_slice_idx++) {
      expanded_builder.SetNext({ptr, slice_shape, type_info.size()});
    }
  }
  assert(expanded_builder.NextSampleIdx() == expanded_batch.num_samples());
}

inline void broadcast_samples(TensorList<GPUBackend> &expanded_batch,
                              const TensorList<GPUBackend> &batch, int num_expanded_samples,
                              const TensorListShape<> &expand_extents,
                              kernels::ScatterGatherGPU &sg, cudaStream_t stream) {
  assert(expand_extents.num_samples() == batch.num_samples());
  const auto &shape = batch.shape();
  auto broadcast_shape = broadcast_sample_shapes(shape, num_expanded_samples, expand_extents);
  expanded_batch.SetLayout(batch.GetLayout());
  expanded_batch.set_order(stream);
  expanded_batch.Resize(broadcast_shape, batch.type());
  auto type_size = batch.type_info().size();
  for (int sample_idx = 0, elem_idx = 0; sample_idx < batch.num_samples(); sample_idx++) {
    auto sample_size = type_size * volume(shape[sample_idx]);
    for (int i = 0; i < volume(expand_extents[sample_idx]); i++) {
      sg.AddCopy(expanded_batch.raw_mutable_tensor(elem_idx++), batch.raw_tensor(sample_idx),
                 sample_size);
    }
  }
  sg.Run(stream, true);
}

inline void broadcast_samples(TensorList<CPUBackend> &expanded_batch,
                              const TensorList<CPUBackend> &batch, int num_expanded_samples,
                              const TensorListShape<> &expand_extents) {
  assert(expand_extents.num_samples() == batch.num_samples());
  const auto &shape = batch.shape();
  auto broadcast_shape = broadcast_sample_shapes(shape, num_expanded_samples, expand_extents);
  expanded_batch.SetLayout(batch.GetLayout());
  expanded_batch.Resize(broadcast_shape, batch.type());
  auto type_size = batch.type_info().size();
  for (int sample_idx = 0, elem_idx = 0; sample_idx < batch.num_samples(); sample_idx++) {
    auto sample_size = type_size * volume(shape[sample_idx]);
    for (int i = 0; i < volume(expand_extents[sample_idx]); i++) {
      std::memcpy(expanded_batch.raw_mutable_tensor(elem_idx++), batch.raw_tensor(sample_idx),
                  sample_size);
    }
  }
}

/** @} */  // end of broadcast_samples

/** @defgroup unfold_outer_dims Utilities to unfold leading extents of the batch.
 * The `ndims_to_unfold` must not exceed the batch's sample_dim.
 * The unfolding may be implemented as a nested for loop:
 * first over samples in the batch and the inner one iterating over the leading extents where each slice becomes
 * a separate sample in the result batch.
 * The resulting batch consists of `batch.shape().first(ndims_to_unfold).num_elements()` samples.
 * The underlying data is not copied, the operation is done in terms of pointers and it is the caller
 * responsibility to make sure the source memory is kept alive for the lifespan of `expanded_batch`.
 * @{
 */

inline TensorListShape<> unfold_outer_dims(const TensorListShape<> &shape, int ndims_to_unfold) {
  if (ndims_to_unfold == 0) {
    return shape;
  } else if (ndims_to_unfold == 1) {
    return unfold_outer_dim(shape);
  } else {
    auto data_shape = collapse_dims(shape, {{0, ndims_to_unfold}});
    return unfold_outer_dim(data_shape);
  }
}

template <typename Backend>
void unfold_outer_dims(TensorVector<Backend> &expanded_batch, const TensorVector<Backend> &batch,
                       int ndims_to_unfold, int num_expanded_samples) {
  auto expanded_builder =
      tv_builder_like(expanded_batch, batch, num_expanded_samples, ndims_to_unfold);
  for (int sample_idx = 0; sample_idx < batch.num_samples(); sample_idx++) {
    auto slices_range = unfolded_slice_range(batch, sample_idx, ndims_to_unfold);
    for (auto &&slice : slices_range) {
      expanded_builder.SetNext(slice);
    }
  }
  assert(expanded_builder.NextSampleIdx() == expanded_batch.num_samples());
}

// TODO(ktokarski) TODO(klecki)
// Rework it when TensorList stops being contigious and supports "true sample" mode
template <typename Backend>
void unfold_outer_dims(TensorList<Backend> &expanded_batch, const TensorList<Backend> &batch,
                       int ndims_to_unfold) {
  const auto &shape = batch.shape();
  expanded_batch.ShareData(batch);
  auto expanded_shape = unfold_outer_dims(shape, ndims_to_unfold);
  expanded_batch.SetLayout(unfolded_sample_layout(batch, ndims_to_unfold));
  expanded_batch.Resize(expanded_shape, batch.type());
}

/** @} */  // end of unfold_outer_dims

inline TensorListShape<> fold_outermost_like(const TensorListShape<> &shape,
                                             const TensorListShape<> &unfolded_extents) {
  assert(shape.num_samples() == unfolded_extents.num_elements());
  if (unfolded_extents.sample_dim() == 0) {
    return shape;
  }
  auto num_samples = unfolded_extents.num_samples();
  TensorListShape<> res(num_samples, unfolded_extents.sample_dim() + shape.sample_dim());
  for (int sample_idx = 0, element_idx = 0; sample_idx < num_samples; sample_idx++) {
    auto group_shape = unfolded_extents[sample_idx];
    auto num_frames = volume(group_shape);
    DALI_ENFORCE(num_frames > 0, make_string("Samples with no frames are not allowed. The ",
                                             sample_idx, " has 0 volume."));
    TensorShape<> element_shape = shape[element_idx++];
    for (int j = 1; j < num_frames; j++) {
      DALI_ENFORCE(element_shape == shape[element_idx++],
                   make_string("Frames in the sample must have equal shapes. Got "
                               "frames of different shape for sample ",
                               sample_idx, "."));
    }
    res.set_tensor_shape(sample_idx, shape_cat(group_shape, element_shape));
  }
  return res;
}
}  // namespace sequence_utils

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_SEQUENCE_SHAPE_H_
