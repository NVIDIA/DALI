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

#include "dali/pipeline/operator/sequence_helper.h"

namespace dali {

std::ostream &operator<<(std::ostream &os, const FrameInfo &frame_info) {
  os << "sample: " << frame_info.sample_idx;
  if (frame_info.frame_idx >= 0) {
    os << ", frame: " << frame_info.frame_idx;
  }
  return os;
}

FrameInfoForSample::FrameInfoForSample(const ExpandDesc *expand_desc)  // NOLINT
    : expand_desc_{expand_desc} {};

// Accessing initial sample indicies defeats the purpose of hiding frames
// processing case from the operator, the utility is expected for exceptional use
// such as reporting errors with clearer error message
FrameInfo FrameInfoForSample::operator()(int flat_sample_idx) const {
  if (!expand_desc_ || !expand_desc_->ShouldExpand()) {
    std::cerr << "Early exit" << std::endl;
    return {flat_sample_idx};
  }
  const auto &shape = expand_desc_->initial_shape_;
  DALI_ENFORCE(0 <= flat_sample_idx && flat_sample_idx < shape.num_elements());
  int sample_idx = 0;
  int frame_offset = flat_sample_idx;
  int sample_num_elements = volume(shape[sample_idx]);
  while (frame_offset - sample_num_elements >= 0) {
    frame_offset -= sample_num_elements;
    sample_num_elements = volume(shape[++sample_idx]);
  }
  if (!expand_desc_->HasChannels()) {
    std::cerr << "no channels exit" << std::endl;
    return {sample_idx, frame_offset};
  } else if (!expand_desc_->HasFrames()) {
    std::cerr << "no frames exit" << std::endl;
    return {sample_idx};
  } else if (expand_desc_->IsChannelFirst()) {
    std::cerr << "channel first exit" << std::endl;
    auto num_frames = shape[sample_idx][expand_desc_->frame_dim_];
    DALI_ENFORCE(num_frames >= 0);
    int frame_idx = frame_offset % num_frames;
    return {sample_idx, frame_idx};
  } else {
    std::cerr << "frames first exit" << std::endl;
    auto num_channels = shape[sample_idx][expand_desc_->channel_dim_];
    DALI_ENFORCE(num_channels >= 0);
    int frame_idx = frame_offset % num_channels;
    return {sample_idx, frame_idx};
  }
}

}  // namespace dali
