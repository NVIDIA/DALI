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

#include <string>
#include <utility>
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/views.h"

namespace dali {

/**
 * @brief Describes expandable prefix of the layout and of the shape of a batch.
 */
class ExpandDesc {
 public:
  inline ExpandDesc(const TensorListShape<> &shape, TensorLayout layout,
                    bool should_expand_channels)
      : layout_{layout},
        frames_dim_{VideoLayoutInfo::FrameDimIndex(layout)},
        channels_dim_{VideoLayoutInfo::ChannelDimIndex(layout)},
        expand_frames_{frames_dim_ == 0 ||
                       (should_expand_channels && frames_dim_ == 1 && channels_dim_ == 0)},
        expand_channels_{should_expand_channels && 0 <= channels_dim_ && channels_dim_ <= 1},
        num_expand_dims_{expand_channels_ + expand_frames_},
        is_channel_first_{num_expand_dims_ == 2 && channels_dim_ < frames_dim_},
        dims_to_expand_{shape.first(num_expand_dims_)},
        num_expanded_{dims_to_expand_.num_elements()} {}


  inline bool ExpandChannels() const {
    return expand_channels_;
  }

  inline bool ExpandFrames() const {
    return expand_frames_;
  }

  inline bool IsChannelFirst() const {
    return is_channel_first_;
  }

  inline int NumDimsToExpand() const {
    return num_expand_dims_;
  }

  inline ptrdiff_t NumExpanded() const {
    return num_expanded_;
  }

  inline ptrdiff_t NumSamples() const {
    return dims_to_expand_.num_samples();
  }

  inline int NumExpanded(int sample_idx) const {
    assert(sample_idx < NumSamples());
    return volume(dims_to_expand_[sample_idx]);
  }

  inline int NumFrames(int sample_idx) const {
    assert(sample_idx < NumSamples());
    assert(frames_dim_ < NumDimsToExpand());
    return dims_to_expand_[sample_idx][frames_dim_];
  }

  inline int NumChannels(int sample_idx) const {
    assert(sample_idx < NumSamples());
    assert(channels_dim_ < NumDimsToExpand());
    return dims_to_expand_[sample_idx][channels_dim_];
  }

  inline const TensorListShape<> &DimsToExpand() const {
    return dims_to_expand_;
  }

  inline TensorLayout Layout() const {
    return layout_;
  }

  inline TensorLayout ExpandedLayout() const {
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
  ptrdiff_t num_expanded_;
};

namespace sequence_utils {

template <typename FrameRange>
class SliceIterator {
  using IndexType = typename FrameRange::IndexType;
  using SliceViewType = typename FrameRange::SliceViewType;

 public:
  SliceIterator(const FrameRange &range, IndexType idx) : range_{range}, idx_{idx} {}

  SliceViewType operator*() const {
    return range_[idx_];
  }

  bool operator==(const SliceIterator &other) {
    return idx_ == other.idx_;
  }

  bool operator!=(const SliceIterator &other) {
    return !(*this == other);
  }

  void operator++() {
    ++idx_;
  }

 private:
  const FrameRange &range_;
  IndexType idx_;
};

struct SliceView {
  uint8_t *ptr;
  TensorShape<> shape;
  size_t type_size;
};

class UnfoldedSliceRange {
 public:
  using IndexType = ptrdiff_t;
  using SliceViewType = SliceView;

  inline UnfoldedSliceRange(SliceViewType view, int ndims_to_unfold)
      : view_{view},
        num_slices_{volume(view.shape.begin(), view.shape.begin() + ndims_to_unfold)},
        slice_shape_{view.shape.begin() + ndims_to_unfold, view.shape.end()},
        slice_stride_{view.type_size * volume(slice_shape_)} {}

  inline SliceIterator<UnfoldedSliceRange> begin() const {
    return {*this, 0};
  }

  inline SliceIterator<UnfoldedSliceRange> end() const {
    return {*this, NumSlices()};
  }

  inline SliceViewType operator[](IndexType idx) const {
    return {view_.ptr + idx * SliceSize(), SliceShape(), view_.type_size};
  }

  inline ptrdiff_t NumSlices() const {
    return num_slices_;
  }

  inline TensorShape<> SliceShape() const {
    return slice_shape_;
  }

  inline size_t SliceSize() const {
    return slice_stride_;
  }

 private:
  SliceViewType view_;
  ptrdiff_t num_slices_;
  TensorShape<> slice_shape_;
  size_t slice_stride_;
};

template <typename Backend>
struct TensorVectorBuilder {
  TensorVectorBuilder(int num_samples, DALIDataType type, bool is_pinned, AccessOrder order)
      : tv_(num_samples) {
    tv_.set_type(type);
    tv_.set_pinned(is_pinned);
    tv_.set_order(order);
  }

  void push(const SliceView &view) {
    tv_[size++].ShareData(view.ptr, view.type_size * volume(view.shape), tv_.is_pinned(),
                          view.shape, tv_.type(), tv_.order());
  }

  TensorVector<Backend> take() {
    assert(size == tv_.num_samples());
    return std::move(tv_);
  }

  TensorVector<Backend> tv_;
  int size = 0;
};

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

template <typename Backend>
TensorList<Backend> unfold_outer_dims(const TensorList<Backend> &data, int ndims_to_unfold) {
  // TODO(ktokarski) TODO(klecki)
  // Rework it when TensorList stops being contigious and supports "true sample" mode
  const auto &shape = data.shape();
  TensorList<Backend> tl;
  tl.ShareData(data);
  auto expanded_shape = unfold_outer_dims(shape, ndims_to_unfold);
  tl.Resize(expanded_shape, data.type());
  return tl;
}

inline TensorListShape<> fold_outermost_like(const TensorListShape<> &shape,
                                             const TensorListShape<> &unfolded_extents) {
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
