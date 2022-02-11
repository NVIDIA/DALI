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

#ifndef DALI_PIPELINE_OPERATOR_SEQUENCE_INFO_H_
#define DALI_PIPELINE_OPERATOR_SEQUENCE_INFO_H_

#include <functional>
#include <ostream>
#include "dali/core/common.h"

namespace dali {

struct FrameInfo {
  int sample_idx;
  int frame_idx = -1;
};

inline std::ostream &operator<<(std::ostream &os, const FrameInfo &frame_info) {
  os << "sample " << frame_info.sample_idx;
  if (frame_info.frame_idx >= 0) {
    os << ", frame " << frame_info.frame_idx;
  }
  return os;
}

using SampleFrameInfoFn = std::function<FrameInfo(int)>;

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_SEQUENCE_INFO_H_
