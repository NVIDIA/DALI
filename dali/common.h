// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_COMMON_H_
#define DALI_COMMON_H_

#ifdef DALI_USE_NVTX
#include "nvToolsExt.h"
#endif

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "dali/api_helper.h"

namespace dali {

// pi
#ifndef M_PI
const float M_PI = 3.14159265358979323846;
#endif

// Using declaration for common types
using std::array;
using std::cout;
using std::endl;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

// Common types
using uint8 = uint8_t;
using int16 = int16_t;
using int64 = int64_t;
using uint64 = uint64_t;
using int32 = int32_t;
using uint32 = uint32_t;

// Basic data type for our indices and dimension sizes
typedef int64_t Index;

enum class OpType {
  GPU = 0,
  CPU = 1,
  MIXED = 2,
  SUPPORT = 3,
  COUNT = 4
};

static std::string to_string(OpType op_type) {
  switch (op_type) {
    case OpType::CPU:
      return "cpu";
    case OpType::GPU:
      return "gpu";
    case OpType::MIXED:
      return "mixed";
    case OpType::SUPPORT:
      return "support";
    default:
      return "<invalid>";
  }
}

struct DALISize {
    int width;
    int height;
};

/**
 * @brief Supported interpolation types
 */
enum DALIInterpType {
  DALI_INTERP_NN = 0,
  DALI_INTERP_LINEAR = 1,
  DALI_INTERP_CUBIC = 2,
  DALI_INTERP_LANCZOS3 = 3,
  DALI_INTERP_TRIANGULAR = 4,
  DALI_INTERP_GAUSSIAN = 5,
};

/**
 * @brief Supported image formats
 */
enum DALIImageType {
  DALI_RGB   = 0,
  DALI_BGR   = 1,
  DALI_GRAY  = 2,
  DALI_YCbCr = 3
};

/**
 * @brief Supported tensor layouts
 */
enum DALITensorLayout {
  DALI_UNKNOWN = -1,
  DALI_NCHW  = 0,
  DALI_NHWC  = 1,
  DALI_NFHWC = 2,
  DALI_NFCHW = 3,
  DALI_SAME  = 4
};

inline bool IsColor(DALIImageType type) {
  return type == DALI_RGB || type == DALI_BGR || type == DALI_YCbCr;
}

inline std::size_t NumberOfChannels(DALIImageType type) {
  return IsColor(type) ? 3 : 1;
}

// Helper to delete copy constructor & copy-assignment operator
#define DISABLE_COPY_MOVE_ASSIGN(name)   \
  name(const name&) = delete;            \
  name& operator=(const name&) = delete; \
  name(name&&) = delete;                 \
  name& operator=(name&&) = delete

// Util to declare anonymous variable
#define CONCAT_1(var1, var2) var1##var2
#define CONCAT_2(var1, var2) CONCAT_1(var1, var2)
#define ANONYMIZE_VARIABLE(name) CONCAT_2(name, __LINE__)

// Basic timerange for profiling
struct TimeRange {
  static const uint32_t kRed = 0xFF0000;
  static const uint32_t kGreen = 0x00FF00;
  static const uint32_t kBlue = 0x0000FF;
  static const uint32_t kYellow = 0xB58900;
  static const uint32_t kOrange = 0xCB4B16;
  static const uint32_t kRed1 = 0xDC322F;
  static const uint32_t kMagenta = 0xD33682;
  static const uint32_t kViolet = 0x6C71C4;
  static const uint32_t kBlue1 = 0x268BD2;
  static const uint32_t kCyan = 0x2AA198;
  static const uint32_t kGreen1 = 0x859900;
  static const uint32_t knvGreen = 0x76B900;

  TimeRange(std::string name, const uint32_t rgb = kBlue) {  // NOLINT
#ifdef DALI_USE_NVTX
    nvtxEventAttributes_t att;
    att.version = NVTX_VERSION;
    att.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    att.colorType = NVTX_COLOR_ARGB;
    att.color = rgb | 0xff000000;
    att.messageType = NVTX_MESSAGE_TYPE_ASCII;
    att.message.ascii = name.c_str();

    nvtxRangePushEx(&att);
    started = true;

#endif
  }

  ~TimeRange() { stop(); }

  void stop() {
#ifdef DALI_USE_NVTX
    if (started) {
      started = false;
      nvtxRangePop();
    }
#endif
  }

#ifdef DALI_USE_NVTX

 private:
  bool started = false;
#endif
};

using std::to_string;

inline std::string to_string(const bool& b) {
  if (b) {
    return "True";
  } else {
    return "False";
  }
}

inline std::string to_string(const DALIInterpType& interpolation) {
  switch (interpolation) {
    case DALI_INTERP_NN:
      return "INTERP_NN";
    case DALI_INTERP_LINEAR:
      return "INTERP_LINEAR";
    case DALI_INTERP_CUBIC:
      return "INTERP_CUBIC";
    default:
      return "<unknown>";
  }
}

inline std::string to_string(const DALIImageType& im_type) {
  switch (im_type) {
    case DALI_RGB:
      return "RGB";
    case DALI_BGR:
      return "BGR";
    case DALI_GRAY:
      return "GRAY";
    case DALI_YCbCr:
      return "YCbCr";
    default:
      return "<unknown>";
  }
}

inline std::string to_string(const DALITensorLayout& layout) {
  switch (layout) {
    case DALI_NCHW:
      return "NCHW";
    case DALI_NHWC:
      return "NHWC";
    case DALI_NFHWC:
      return "NFHWC";
    case DALI_NFCHW:
      return "NFCHW";
    case DALI_SAME:
      return "SAME";
    default:
      return "<unknown>";
  }
}

inline DALITensorLayout GetElementLayout(DALITensorLayout sequence_layout) {
  switch (sequence_layout) {
    case DALI_NFHWC:
      return DALI_NHWC;
    case DALI_NFCHW:
      return DALI_NCHW;
    default:  // if cannot produce anything meaningful, keep the same layout
      return sequence_layout;
  }
}

inline DALITensorLayout GetSequenceLayout(DALITensorLayout element_layout) {
  switch (element_layout) {
    case DALI_NHWC:
      return DALI_NFHWC;
    case DALI_NCHW:
      return DALI_NFCHW;
    default:  // if cannot produce anything meaningful, keep the same layout
      return element_layout;
  }
}

inline bool IsSequence(DALITensorLayout layout) {
  return layout == DALI_NFHWC || layout == DALI_NFCHW;
}

template <typename T>
auto to_string(const T& v) -> decltype(std::string(v)) {
  return v;
}

template <typename T>
auto to_string(const T& t) -> decltype(t.ToString()) {
  return t.ToString();
}

template <typename T>
std::string to_string(const std::vector<T>& v) {
  std::string ret = "[";
  for (T t : v) {
    ret += to_string(t);
    ret += ", ";
  }
  ret += "]";
  return ret;
}

template <typename T>
struct is_vector : std::false_type {};

template <typename T, typename A>
struct is_vector<std::vector<T, A> > : std::true_type {};

template <typename T>
struct is_std_array : std::false_type {};

template <typename T, size_t A>
struct is_std_array<std::array<T, A> > : std::true_type {};

std::vector<std::string> string_split(const std::string &s, const char delim);

template <bool Value, typename Type = void>
using enable_if_t = typename std::enable_if<Value, Type>::type;

}  // namespace dali

#define LOG_LINE \
  if (0) \
  std::cout << __FILE__ << ":" << __LINE__ << ": "

#define ERROR_LOG \
  if (1) \
  std::cerr << __FILE__ << ":" << __LINE__ << ": "

#endif  // DALI_COMMON_H_
