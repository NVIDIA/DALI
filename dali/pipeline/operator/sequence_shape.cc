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

#include "dali/pipeline/operator/sequence_shape.h"

namespace dali {

FrameInfo ExpandDescFrameInfoFn::operator()(int flat_sample_idx) const {
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

}  // namespace dali
