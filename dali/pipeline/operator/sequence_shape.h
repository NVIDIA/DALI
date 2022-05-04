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
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/views.h"

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

template <typename Range>
class RangeIterator {
  using IndexType = typename Range::IndexType;
  using ValueType = typename Range::ValueType;

 public:
  RangeIterator(const Range &range, IndexType idx) : range_{range}, idx_{idx} {}

  ValueType operator*() const {
    return range_[idx_];
  }

  bool operator==(const RangeIterator &other) const {
    return idx_ == other.idx_;
  }

  bool operator!=(const RangeIterator &other) const {
    return !(*this == other);
  }

  void operator++() {
    ++idx_;
  }

 private:
  const Range &range_;
  IndexType idx_;
};

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


/**
 * @brief Utility to build the sharing tensor vector from slices of samples of
 * another tensor vector.
 */
template <typename Backend>
struct TensorVectorBuilder {
  TensorVectorBuilder(int num_samples, DALIDataType type, int sample_dim, TensorLayout layout,
                      bool is_pinned, AccessOrder order)
      : tv_(num_samples) {
    tv_.set_type(type);
    tv_.set_sample_dim(sample_dim);
    tv_.SetLayout(layout);
    tv_.set_pinned(is_pinned);
    tv_.set_order(order);
  }

  void push(const SliceView &view) {
    std::shared_ptr<void> ptr(view.ptr, [](void *) {});  // no deleter
    tv_.UnsafeSetSample(size++, ptr, view.type_size * volume(view.shape), tv_.is_pinned(),
                        view.shape, tv_.type(), tv_.order(), tv_.GetLayout());
  }

  /**
   * @brief Returns the created tensor vector: must be called after pusing exactly ``num_samples``.
   * Calling ``take`` invalidates the ``TensorVectorBuilder`` instance.
   */
  TensorVector<Backend> take() {
    assert(size == tv_.num_samples());
    return std::move(tv_);
  }

  TensorVector<Backend> tv_;
  int size = 0;
};


template <typename Backend>
TensorVectorBuilder<Backend> tv_builder_like(const TensorVector<Backend> &data, int num_samples,
                                             int ndims_to_unfold = 0) {
  assert(data.sample_dim() >= ndims_to_unfold);
  auto sample_dim = data.sample_dim() - ndims_to_unfold;
  const auto &initial_layout = data.GetLayout();
  TensorLayout layout = initial_layout.empty() ? "" : initial_layout.last(sample_dim);
  return {num_samples, data.type(), sample_dim, layout, data.is_pinned(), data.order()};
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

inline TensorListShape<> broadcast_samples(const TensorListShape<> &shape,
                                           const ExpandDesc &expand_desc) {
  assert(shape.num_samples() == expand_desc.NumSamples());
  TensorListShape<> broadcast(expand_desc.NumExpanded(), shape.sample_dim());
  for (int sample_idx = 0, elem_idx = 0; sample_idx < expand_desc.NumSamples(); sample_idx++) {
    const auto &sample_shape = shape[sample_idx];
    for (int i = 0; i < expand_desc.NumExpanded(sample_idx); i++) {
      broadcast.set_tensor_shape(elem_idx++, sample_shape);
    }
  }
  return broadcast;
}

template <typename Backend>
TensorList<Backend> unfold_outer_dims(const TensorList<Backend> &data, int ndims_to_unfold) {
  // TODO(ktokarski) TODO(klecki)
  // Rework it when TensorList stops being contigious and supports "true sample" mode
  const auto &shape = data.shape();
  TensorList<Backend> tl;
  tl.ShareData(data);
  auto expanded_shape = unfold_outer_dims(shape, ndims_to_unfold);
  tl.Resize(expanded_shape, data.type());
  const auto &initial_layout = data.GetLayout();
  TensorLayout layout = initial_layout.empty() ? "" : initial_layout.sub(ndims_to_unfold);
  tl.SetLayout(layout);
  return tl;
}

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
