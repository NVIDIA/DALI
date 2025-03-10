// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_VIDEO_VIDEO_READER_UTILS_H_
#define DALI_OPERATORS_VIDEO_VIDEO_READER_UTILS_H_

#include <dirent.h>
#include <string>
#include <vector>
#include "dali/core/error_handling.h"
#include "dali/core/boundary.h"
#include "libavutil/rational.h"
#include "dali/pipeline/operator/op_spec.h"

namespace dali {

struct VideoFileMeta {
  std::string video_file;
  int label;
  float start_time;
  float end_time;
  bool operator<(const VideoFileMeta& right) {
    return video_file < right.video_file;
  }
};

inline float TimestampToSeconds(AVRational timebase, int64_t timestamp) {
  return static_cast<float>(
    static_cast<double>(timestamp) * timebase.num / timebase.den);
}

inline int64_t SecondsToTimestamp(AVRational timebase, float seconds) {
  return static_cast<int64_t>(
    static_cast<double>(seconds) * timebase.den / timebase.num);
}

std::vector<VideoFileMeta> GetVideoFiles(const std::string& file_root,
                                         const std::vector<std::string>& filenames, bool use_labels,
                                         const std::vector<int>& labels,
                                         const std::string& file_list);

inline boundary::BoundaryType GetBoundaryType(const OpSpec &spec) {
  auto pad_mode_str = spec.template GetArgument<std::string>("pad_mode");
  boundary::BoundaryType boundary_type = boundary::BoundaryType::ISOLATED;
  if (pad_mode_str == "none" || pad_mode_str == "") {
    boundary_type = boundary::BoundaryType::ISOLATED;
  } else if (pad_mode_str == "constant") {
    boundary_type = boundary::BoundaryType::CONSTANT;
  } else if (pad_mode_str == "edge" || pad_mode_str == "repeat") {
    boundary_type = boundary::BoundaryType::CLAMP;
  } else if (pad_mode_str == "reflect_1001" || pad_mode_str == "symmetric") {
    boundary_type = boundary::BoundaryType::REFLECT_1001;
  } else if (pad_mode_str == "reflect_101" || pad_mode_str == "reflect") {
    boundary_type = boundary::BoundaryType::REFLECT_101;
  } else {
    DALI_FAIL(make_string("Invalid pad_mode: ", pad_mode_str, "\n",
                          "Valid options are: none, constant, edge, reflect_1001, reflect_101"));
  }
  return boundary_type;
}

template <typename Backend>
const uint8_t* ConstantFrame(Tensor<Backend>& constant_frame, const TensorShape<>& shape,
                             span<const uint8_t> fill_value, cudaStream_t stream,
                             bool reuse_existing_data = true) {
  if (reuse_existing_data && constant_frame.shape().num_elements() >= shape.num_elements()) {
    return constant_frame.template data<uint8_t>();
  }
  if (std::is_same_v<Backend, CPUBackend>) {
    constant_frame.set_pinned(false);
  }
  constant_frame.Resize(shape, DALI_UINT8);
  DALI_ENFORCE(fill_value.size() == 1 || static_cast<int>(fill_value.size()) == shape[2],
               make_string("Fill value size must be 1 or equal to the number of channels. Got ",
                           fill_value.size(), " with num_channels=", shape[2]));

  auto fill_data = [&](uint8_t* data) {
    if (fill_value.size() == 1) {
      std::fill(data, data + shape.num_elements(), fill_value[0]);
    } else {
      for (int i = 0; i < shape.num_elements(); i++) {
        data[i] = fill_value[i % fill_value.size()];
      }
    }
  };

  if (std::is_same_v<Backend, GPUBackend>) {
    Tensor<CPUBackend> tmp;
    tmp.set_pinned(true);
    tmp.Resize(shape, DALI_UINT8);
    fill_data(tmp.template mutable_data<uint8_t>());
    constant_frame.Copy(tmp, stream);
  } else {
    fill_data(constant_frame.template mutable_data<uint8_t>());
  }
  return constant_frame.template data<uint8_t>();
}

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_VIDEO_READER_UTILS_H_
