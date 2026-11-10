// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
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
  double start;
  double end;
  int start_frame = -1;
  int end_frame = -1;
  bool operator<(const VideoFileMeta& right) {
    return filename < right.filename;
  }
};

inline double TimestampToSeconds(AVRational timebase, int64_t timestamp) {
  return static_cast<double>(timestamp) * timebase.num / timebase.den;
}

namespace detail {

/// Largest magnitude SecondsToTimestamp returns: 2^62 ticks, which no real stream reaches, and
/// which leaves enough headroom for callers to add a pts offset without overflowing int64_t.
constexpr double kMaxTimestampTicks = 4611686018427387904.0;  // 2^62

/// Converts a whole number of ticks to int64_t, saturating to +-kMaxTimestampTicks -- a plain
/// cast is UB for values outside of int64_t's range (e.g. a file_list time of 1e20 s).
inline int64_t SaturateTicks(double ticks) {
  if (ticks >= kMaxTimestampTicks)
    return static_cast<int64_t>(kMaxTimestampTicks);
  if (ticks <= -kMaxTimestampTicks)
    return -static_cast<int64_t>(kMaxTimestampTicks);
  return static_cast<int64_t>(ticks);
}

}  // namespace detail

/**
 * @brief Converts a time in seconds to a timestamp in `timebase` units, rounding down or up.
 *
 * The result saturates to +-2^62 ticks for times too large to represent, so a huge start time
 * is still detected as being past the end of the stream, and a huge end time clamps to the end.
 */
inline int64_t SecondsToTimestamp(AVRational timebase, double seconds, bool round_down) {
  if (std::isnan(seconds))
    throw std::invalid_argument("Cannot convert a NaN time to a timestamp.");
  // pts are integers, so floor(x) >= pts <=> x >= pts and ceil(x) <= pts <=> x <= pts:
  // quantizing in the direction of the lookup keeps GetFrameIdxByTimestamp exact. Rounding to the
  // nearest tick would not: a time just above the pts of a frame can round down to it.
  //
  // Values within a relative 1e-12 of an integer are floating-point noise (e.g. 1.001 s at
  // 1/1000 is 1000.9999999999999 ticks) and are snapped first. 1e-12 is several thousand times
  // the error of the conversion (a few 1e-16) and stays below half a tick up to ~5e11 ticks.
  double ticks = seconds * timebase.den / timebase.num;
  constexpr double kRelativeEpsilon = 1e-12;
  double rounded = std::round(ticks);
  double epsilon = std::max(1.0, std::abs(rounded)) * kRelativeEpsilon;
  if (std::abs(ticks - rounded) < epsilon) {
    ticks = rounded;
  }
  return detail::SaturateTicks(round_down ? std::floor(ticks) : std::ceil(ticks));
}

DLL_PUBLIC std::vector<VideoFileMeta> GetVideoFiles(
    const std::string& file_root, const std::vector<std::string>& filenames, bool use_labels,
    const std::vector<int>& labels, const std::string& file_list);

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

template <typename Backend>
const uint8_t* ConstantFrame(Tensor<Backend>& constant_frame, const TensorShape<>& shape,
                             span<const uint8_t> fill_value, cudaStream_t stream,
                             bool reuse_existing_data = true) {
  if (reuse_existing_data && constant_frame.shape().num_elements() >= shape.num_elements()) {
    return constant_frame.template data<uint8_t>();
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

std::string av_error_string(int ret);

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_VIDEO_READER_UTILS_H_
