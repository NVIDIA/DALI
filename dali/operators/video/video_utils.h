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

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}

#include <dirent.h>
#include <string>
#include <type_traits>
#include <vector>
#include "dali/core/error_handling.h"
#include "dali/core/boundary.h"
#include "libavutil/rational.h"
#include "dali/pipeline/operator/op_spec.h"

namespace dali {

using AVPacketScope = std::unique_ptr<AVPacket, decltype(&av_packet_unref)>;

struct VideoFileMeta {
  std::string filename;
  int label;
  float start;
  float end;
  int start_frame = -1;
  int end_frame = -1;
  bool operator<(const VideoFileMeta& right) {
    return filename < right.filename;
  }
};

inline double TimestampToSeconds(AVRational timebase, int64_t timestamp) {
  return static_cast<double>(timestamp) * timebase.num / timebase.den;
}

inline int64_t SecondsToTimestamp(AVRational timebase, double seconds) {
  return static_cast<int64_t>(seconds * timebase.den / timebase.num);
}

std::vector<VideoFileMeta> GetVideoFiles(const std::string& file_root,
                                         const std::vector<std::string>& filenames, bool use_labels,
                                         const std::vector<int>& labels,
                                         const std::string& file_list);

enum class FrameNumPolicy {
  None,      // no frame number output
  Scalar,    // first frame index as a scalar with shape (1,)
  Sequence   // per-frame indices with shape (F,); padded frames get -1
};

inline FrameNumPolicy ParseFrameNumPolicy(const std::string &s) {
  // "True"/"False" are the Python str(bool) representations, kept for backward compatibility
  // with code that passes enable_frame_num=True/False (Python bools).
  if (s == "none" || s == "False")   return FrameNumPolicy::None;
  if (s == "scalar" || s == "True")  return FrameNumPolicy::Scalar;
  if (s == "sequence")               return FrameNumPolicy::Sequence;
  DALI_FAIL(make_string("Invalid enable_frame_num value: '", s,
                        "'. Valid values are: 'none', 'scalar', 'sequence'."));
}

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

/**
 * @brief Returns a device/host buffer holding one frame of `shape` filled with `fill_value`.
 *
 * `fill_value` is expressed in the 8-bit [0, 255] scale (one value, or one per channel). For
 * DALI_FLOAT it is converted like decoded pixels are: kept as is, or divided by 255 when
 * `normalized` is set. The buffer is reused when it already has the right type and is large
 * enough.
 */
template <typename Backend>
const uint8_t* ConstantFrame(Tensor<Backend>& constant_frame, const TensorShape<>& shape,
                             span<const uint8_t> fill_value, cudaStream_t stream,
                             bool reuse_existing_data = true, DALIDataType dtype = DALI_UINT8,
                             bool normalized = false) {
  DALI_ENFORCE(dtype == DALI_UINT8 || dtype == DALI_FLOAT,
               make_string("Unsupported constant frame type: ", dtype));
  if (reuse_existing_data && constant_frame.type() == dtype &&
      constant_frame.shape().num_elements() >= shape.num_elements()) {
    return static_cast<const uint8_t*>(constant_frame.raw_data());
  }
  DALI_ENFORCE(fill_value.size() == 1 || static_cast<int>(fill_value.size()) == shape[2],
               make_string("Fill value size must be 1 or equal to the number of channels. Got ",
                           fill_value.size(), " with num_channels=", shape[2]));

  auto fill_data = [&](auto* data) {
    using T = std::remove_pointer_t<decltype(data)>;
    for (int64_t i = 0; i < shape.num_elements(); i++) {
      uint8_t v = fill_value[fill_value.size() == 1 ? 0 : i % fill_value.size()];
      if constexpr (std::is_same_v<T, float>) {
        data[i] = normalized ? v / 255.0f : static_cast<float>(v);
      } else {
        data[i] = v;
      }
    }
  };
  auto fill_host_tensor = [&](Tensor<CPUBackend>& tensor) {
    if (dtype == DALI_FLOAT)
      fill_data(tensor.template mutable_data<float>());
    else
      fill_data(tensor.template mutable_data<uint8_t>());
  };

  constant_frame.Resize(shape, dtype);
  if constexpr (std::is_same_v<Backend, GPUBackend>) {
    Tensor<CPUBackend> tmp;
    tmp.set_pinned(true);
    tmp.Resize(shape, dtype);
    fill_host_tensor(tmp);
    constant_frame.Copy(tmp, stream);
  } else {
    fill_host_tensor(constant_frame);
  }
  return static_cast<const uint8_t*>(constant_frame.raw_data());
}

std::string av_error_string(int ret);

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_VIDEO_READER_UTILS_H_
