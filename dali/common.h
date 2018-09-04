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

#include <cuda_runtime_api.h>  // for __align__ & CUDART_VERSION
#include <cuda_fp16.h>  // for __half & related methods
#include <cuda_profiler_api.h>

#ifdef DALI_USE_NVTX
#include "nvToolsExt.h"
#endif

#include <cstdint>
#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "dali/api_helper.h"

namespace dali {

// pi
#ifndef M_PI
const float M_PI =  3.14159265358979323846;
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
typedef uint8_t uint8;
typedef int16_t int16;
typedef int64_t int64;
typedef uint64_t uint64;
typedef int32_t int32;
typedef uint32_t uint32_t;

// Basic data type for our indices and dimension sizes
typedef int64_t Index;

// Only supported on the GPU
typedef __half float16;

/**
 * @brief Supported interpolation types
 */
enum DALIInterpType {
  DALI_INTERP_NN = 0,
  DALI_INTERP_LINEAR = 1,
  DALI_INTERP_CUBIC = 2
};

/**
 * @brief Supported image formats
 */
enum DALIImageType {
  DALI_RGB = 0,
  DALI_BGR = 1,
  DALI_GRAY = 2
};

/**
 * @brief Supported tensor layouts
 */
enum DALITensorLayout {
  DALI_NCHW = 0,
  DALI_NHWC = 1,
  DALI_SAME = 2
};

inline bool IsColor(DALIImageType type) {
  if ((type == DALI_RGB) || (type == DALI_BGR)) {
    return true;
  }
  return false;
}

// Compatible wrapper for CUDA 8 which does not have builtin static_cast<float16>
template<typename dst>
__device__ inline dst StaticCastGpu(float val)
{ return static_cast<dst>(val); }

#if defined(__CUDACC__) && defined(CUDART_VERSION) && CUDART_VERSION < 9000
template<>
__device__ inline float16 StaticCastGpu(float val)
{ return __float2half(static_cast<float>(val)); }
#endif  // defined(CUDART_VERSION) && CUDART_VERSION < 9000

// Helper to delete copy constructor & copy-assignment operator
#define DISABLE_COPY_MOVE_ASSIGN(name)          \
  name(const name&) = delete;                   \
  name& operator=(const name&) = delete;        \
  name(name&&) = delete;                        \
  name& operator=(name&&) = delete

// Util to declare anonymous variable
#define CONCAT_1(var1, var2) var1##var2
#define CONCAT_2(var1, var2) CONCAT_1(var1, var2)
#define ANONYMIZE_VARIABLE(name) CONCAT_2(name, __LINE__)

// Starts profiling DALI
inline void DALIProfilerStart() {
  cudaProfilerStart();
}

inline void DALIProfilerStop() {
  cudaProfilerStop();
}

// Basic timerange for profiling
struct TimeRange {
  static const uint32_t kRed     = 0xFF0000;
  static const uint32_t kGreen   = 0x00FF00;
  static const uint32_t kBlue    = 0x0000FF;
  static const uint32_t kYellow  = 0xB58900;
  static const uint32_t kOrange  = 0xCB4B16;
  static const uint32_t kRed1    = 0xDC322F;
  static const uint32_t kMagenta = 0xD33682;
  static const uint32_t kViolet  = 0x6C71C4;
  static const uint32_t kBlue1   = 0x268BD2;
  static const uint32_t kCyan    = 0x2AA198;
  static const uint32_t kGreen1  = 0x859900;

  TimeRange(std::string name, const uint32_t rgb = kBlue) { // NOLINT
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

  ~TimeRange() {
    stop();
  }

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
    default:
      return "<unknown>";
  }
}

template<typename T>
auto to_string(const T& v)
  -> decltype(std::string(v)) {
    return v;
}

template<typename T>
auto to_string(const T& t)
  -> decltype(t.ToString()) {
  return t.ToString();
}

template<typename T>
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
struct is_vector: std::false_type {};

template <typename T, typename A>
struct is_vector<std::vector<T, A> >: std::true_type {};

template <typename T>
struct is_array: std::false_type {};

template <typename T, size_t A>
struct is_array<std::array<T, A> >: std::true_type {};

}  // namespace dali

#endif  // DALI_COMMON_H_
