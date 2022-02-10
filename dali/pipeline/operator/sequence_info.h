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

// Accessing initial sample indicies defeats the purpose of hiding frames
// handling from the operator, the utility is expected for exceptional use
// such as reporting errors to user with clearer message
struct DLL_PUBLIC SampleFrameInfoFn {
  DLL_PUBLIC inline SampleFrameInfoFn(std::function<FrameInfo(int)> frame_info_fn = nullptr)
      : frame_info_fn_{frame_info_fn} {}

  DLL_PUBLIC inline FrameInfo operator()(int flat_sample_idx) const {
    if (!frame_info_fn_) {
      return {flat_sample_idx};
    }
    return frame_info_fn_(flat_sample_idx);
  }

 private:
  std::function<FrameInfo(int)> frame_info_fn_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_SEQUENCE_INFO_H_
