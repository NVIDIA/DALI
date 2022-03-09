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
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/views.h"

namespace dali {

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

template <typename Backend>
TensorVector<Backend> unfold_outer_dims(const TensorVector<Backend> &data, int ndims_to_unfold) {
  const auto &shape = data.shape();
  auto group_shapes = shape.first(ndims_to_unfold);
  const auto num_unfolded_samples = group_shapes.num_elements();
  TensorVector<Backend> unfolded_tensor(num_unfolded_samples);
  auto type_info = data.type_info();
  auto type = type_info.id();
  auto is_pinned = data.is_pinned();
  auto order = data.order();
  unfolded_tensor.set_type(type);
  unfolded_tensor.set_pinned(is_pinned);
  unfolded_tensor.set_order(order);
  int elem_idx = 0;
  for (int sample_idx = 0; sample_idx < shape.num_samples(); sample_idx++) {
    const auto &sample_shape = shape[sample_idx];
    int num_frames = volume(sample_shape.begin(), sample_shape.begin() + ndims_to_unfold);
    TensorShape<> element_shape{sample_shape.begin() + ndims_to_unfold, sample_shape.end()};
    int element_volume = volume(element_shape);
    auto num_bytes = type_info.size() * element_volume;
    uint8_t *base_ptr =
        const_cast<uint8_t *>(static_cast<const uint8_t *>(data.raw_tensor(sample_idx)));
    for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
      uint8_t *ptr = base_ptr + frame_idx * num_bytes;
      unfolded_tensor[elem_idx++].ShareData(ptr, num_bytes, is_pinned, element_shape, type, order);
    }
  }
  assert(elem_idx == num_unfolded_samples);
  return unfolded_tensor;
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

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_SEQUENCE_SHAPE_H_
