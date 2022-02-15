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
#include <type_traits>
#include <utility>
#include <vector>
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/sequence_info.h"


namespace dali {

struct LayoutDesc {
  static inline LayoutDesc FrameAndChannel(const TensorLayout &layout) {
    int frame_dim = VideoLayoutInfo::FrameDimIndex(layout);
    int channel_dim = ImageLayoutInfo::ChannelDimIndex(layout);
    frame_dim = frame_dim > 1 ? -1 : frame_dim;
    channel_dim = channel_dim > 1 ? -1 : channel_dim;
    if (frame_dim != 0 && channel_dim != 0) {
      return {};
    }
    return {frame_dim, channel_dim};
  }

  static inline LayoutDesc Frame(const TensorLayout &layout) {
    if (VideoLayoutInfo::FrameDimIndex(layout) != 0) {
      return {};
    }
    return {0};
  }

  inline int NumDims() const {
    return (frame_dim >= 0) + (channel_dim >= 0);
  }

  int frame_dim = -1;
  int channel_dim = -1;
};

class ExpandDesc {
 public:
  inline ExpandDesc() = default;
  inline ExpandDesc(const TensorListShape<> &shape, const TensorLayout &layout, bool should_expand)
      : layout_desc_{([should_expand, &shape, &layout]() {
          DALI_ENFORCE(layout.empty() || shape.sample_dim() == layout.size());
          return should_expand ? LayoutDesc::FrameAndChannel(layout) : LayoutDesc::Frame(layout);
        })()},
        has_channels_{layout_desc_.channel_dim >= 0},
        has_frames_{layout_desc_.frame_dim >= 0},
        is_channel_first_{has_channels_ && layout_desc_.channel_dim < layout_desc_.frame_dim},
        num_dims_{has_channels_ + has_frames_},
        expanded_layout_{layout.first(num_dims_)},
        expanded_shape_{shape.first(num_dims_)},
        num_elements_{expanded_shape_.num_elements()} {}

  inline int NumDims() const {
    return num_dims_;
  }

  inline ptrdiff_t NumElements() const {
    return num_elements_;
  }

  inline int NumSamples() const {
    return expanded_shape_.num_samples();
  }

  inline bool HasChannels() const {
    return has_channels_;
  }

  inline bool HasFrames() const {
    return has_frames_;
  }

  inline bool IsChannelFirst() const {
    return is_channel_first_;
  }

  inline int NumFrames(int sample_idx) const {
    return expanded_shape_[sample_idx][layout_desc_.frame_dim];
  }

  inline int NumChannels(int sample_idx) const {
    return expanded_shape_[sample_idx][layout_desc_.channel_dim];
  }

  inline int NumElements(int sample_idx) const {
    return volume(expanded_shape_[sample_idx]);
  }

  inline TensorShape<> ExpandedShape(int sample_idx) const {
    return expanded_shape_[sample_idx];
  }

  inline TensorLayout ExpandedLayout() const {
    return expanded_layout_;
  }

 private:
  LayoutDesc layout_desc_;
  bool has_channels_;
  bool has_frames_;
  bool is_channel_first_;
  int num_dims_;
  TensorLayout expanded_layout_;
  TensorListShape<> expanded_shape_;
  ptrdiff_t num_elements_;
};


class ExpandDescFrameInfoFn {
 public:
  ExpandDescFrameInfoFn(const ExpandDesc *expand_desc)  // NOLINT
      : expand_desc_{expand_desc} {};

  inline FrameInfo operator()(int flat_sample_idx) const {
    DALI_ENFORCE(expand_desc_);
    const auto &expand_desc = *expand_desc_;
    if (expand_desc.NumDims() == 0) {
      return {flat_sample_idx};
    }
    DALI_ENFORCE(0 <= flat_sample_idx && flat_sample_idx < expand_desc.NumElements());
    int sample_idx = 0;
    int frame_offset = flat_sample_idx;
    int sample_num_elements = expand_desc.NumElements(sample_idx);
    while (frame_offset - sample_num_elements >= 0) {
      frame_offset -= sample_num_elements;
      sample_num_elements = expand_desc.NumElements(++sample_idx);
    }
    if (!expand_desc.HasChannels()) {
      return {sample_idx, frame_offset};
    } else if (!expand_desc.HasFrames()) {
      return {sample_idx};
    } else if (expand_desc.IsChannelFirst()) {
      auto num_frames = expand_desc.NumFrames(sample_idx);
      DALI_ENFORCE(num_frames > 0);
      int frame_idx = frame_offset % num_frames;
      return {sample_idx, frame_idx};
    } else {
      auto num_channels = expand_desc.NumChannels(sample_idx);
      DALI_ENFORCE(num_channels > 0);
      int frame_idx = frame_offset / num_channels;
      return {sample_idx, frame_idx};
    }
  }

 private:
  const ExpandDesc *expand_desc_;
};

template <typename Backend>
TensorVector<Backend> unfold_outer_dims(const TensorVector<Backend> &data, int num_unfold_dims) {
  const auto &initial_shape = data.shape();
  auto unfold_shape = initial_shape.first(num_unfold_dims);
  const auto target_num_samples = unfold_shape.num_elements();
  TensorVector<Backend> flat_tensor(target_num_samples);
  int elem_idx = 0;
  auto type_info = data.type_info();
  auto type = type_info.id();
  auto is_pinned = data.is_pinned();
  auto order = data.order();
  flat_tensor.set_type(type);
  flat_tensor.set_pinned(is_pinned);
  flat_tensor.set_order(order);
  for (int sample_idx = 0; sample_idx < initial_shape.num_samples(); sample_idx++) {
    const auto &sample_shape = initial_shape[sample_idx];
    int num_frames = volume(sample_shape.begin(), sample_shape.begin() + num_unfold_dims);
    TensorShape<> element_shape{sample_shape.begin() + num_unfold_dims, sample_shape.end()};
    int element_volume = volume(element_shape);
    auto num_bytes = type_info.size() * element_volume;
    uint8_t *base_ptr;
    base_ptr = const_cast<uint8_t *>(static_cast<const uint8_t *>(data.raw_tensor(sample_idx)));
    for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
      uint8_t *ptr = base_ptr + frame_idx * num_bytes;
      flat_tensor[elem_idx++].ShareData(ptr, num_bytes, is_pinned, element_shape, type, order);
    }
  }
  DALI_ENFORCE(elem_idx == target_num_samples, "TODO: change me to assert");
  return flat_tensor;
}

template <typename Backend>
TensorList<Backend> unfold_outer_dims(const TensorList<Backend> &data, int num_unfold_dims) {
  TensorList<Backend> tl;
  tl.ShareData(data);
  auto shape = unfold_outer_dims(data.shape(), num_unfold_dims);
  tl.Resize(shape, data.type());
  return tl;
}

inline TensorListShape<> unfold_outer_dims(const TensorListShape<> &shape, int num_unfold_dims) {
  DALI_ENFORCE(num_unfold_dims >= 0 && num_unfold_dims < shape.sample_dim(),
               "TODO: maybe just an assert?");
  if (num_unfold_dims == 0) {
    return shape;
  } else if (num_unfold_dims == 1) {
    return unfold_outer_dim(shape);
  } else {
    auto data_shape = collapse_dims(shape, {{0, num_unfold_dims}});
    return unfold_outer_dim(data_shape);
  }
}

inline TensorListShape<> fold_outermost_like(const TensorListShape<> &shape,
                                             const ExpandDesc &expand_desc) {
  if (expand_desc.NumDims() == 0) {
    return shape;
  }
  auto num_samples = shape.num_samples();
  auto num_elements = expand_desc.NumElements();
  auto num_groups = expand_desc.NumSamples();
  DALI_ENFORCE(num_samples == num_elements);
  TensorListShape<> res(num_groups, expand_desc.NumDims() + shape.sample_dim());
  int sample_offset = 0;
  for (int i = 0; i < num_groups; i++) {
    const auto &frame_shape = shape[sample_offset];
    auto expanded_shape = expand_desc.ExpandedShape(i);
    res.set_tensor_shape(i, shape_cat(expanded_shape, frame_shape));
    auto num_frames = volume(expanded_shape);
    for (int j = 1; j < num_frames; j++) {
      DALI_ENFORCE(shape[sample_offset + j] == frame_shape);
    }
    sample_offset += num_frames;
  }
  return res;
}

inline TensorVector<CPUBackend> expand_argument_like(const TensorVector<CPUBackend> &tv,
                                                     const std::string &name,
                                                     const ExpandDesc &expand_desc) {
  const auto &layout = tv.GetLayout();
  bool arg_has_frames_dim = VideoLayoutInfo::FrameDimIndex(layout) == 0;
  DALI_ENFORCE(
      !arg_has_frames_dim || expand_desc.HasFrames(),
      "Argument input contains frames but the input of the operator does not. How ridiculous.");
  auto num_elements = expand_desc.NumElements();
  int num_samples = expand_desc.NumSamples();
  TensorVector<CPUBackend> flat_tensor(num_elements);
  int arg_sample_dim = tv.sample_dim();
  DALI_ENFORCE(!arg_has_frames_dim || arg_sample_dim >= 1,
               make_string("The layout of tensor argument ", name,
                           "contains frames, but it has 0 dimensionality."));
  int sample_offset = 0;
  for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    const auto &arg_tensor = tv[sample_idx];
    const auto &arg_shape = arg_tensor.shape();
    int num_input_frames = expand_desc.HasFrames() ? expand_desc.NumFrames(sample_idx) : 1;
    int num_arg_frames = !arg_has_frames_dim ? 1 : arg_shape[0];
    auto elem_shape = arg_has_frames_dim ? arg_shape.last(arg_sample_dim - 1) : arg_shape;
    DALI_ENFORCE(
        num_arg_frames == 1 || num_input_frames == num_arg_frames,
        make_string("The tensor argument ", name, " for sample ", sample_idx,
                    " for an input that contains frames, should either be a single set "
                    "of parameters to be reused accross all frames in the sample all match the "
                    "number of frames. Got ",
                    num_arg_frames, " but there are ", num_input_frames, " frames in the sample."));
    uint8_t *base_ptr = const_cast<uint8_t *>(static_cast<const uint8_t *>(arg_tensor.raw_data()));
    auto elem_volume = volume(elem_shape);
    auto type_info = arg_tensor.type_info();
    auto num_bytes = type_info.size() * elem_volume;
    auto type = type_info.id();
    auto is_pinned = arg_tensor.is_pinned();
    auto order = arg_tensor.order();
    int num_input_channels = expand_desc.HasChannels() ? expand_desc.NumChannels(sample_idx) : 1;
    int num_repeat =
        num_arg_frames == 1 ? num_input_channels * num_input_frames : num_input_channels;
    int inner_stride, outer_stride;
    if (num_arg_frames == 1) {
      inner_stride = 1;
      outer_stride = 1;
    } else if (expand_desc.IsChannelFirst()) {
      inner_stride = num_input_frames;
      outer_stride = 1;
    } else {
      inner_stride = 1;
      outer_stride = num_input_channels;
    }
    for (int i = 0; i < num_arg_frames; i++) {  // expands argument
      uint8_t *ptr = base_ptr + i * num_bytes;
      for (int j = 0; j < num_repeat; j++) {  // repeats argument
        flat_tensor[sample_offset + i * outer_stride + j * inner_stride].ShareData(
            ptr, num_bytes, is_pinned, elem_shape, type, order);
      }
    }
    sample_offset += num_input_channels * num_input_frames;
  }
  DALI_ENFORCE(sample_offset == num_elements);
  return flat_tensor;
}

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_SEQUENCE_SHAPE_H_
