// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_COMMON_H_
#define NDLL_COMMON_H_

#include <cuda_fp16.h>  // for __half & related methods
#include <cuda_profiler_api.h>

#ifdef NDLL_USE_NVTX
#include "nvToolsExt.h"
#endif

#include <cstdint>
#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace ndll {

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
enum NDLLInterpType {
  NDLL_INTERP_NN = 0,
  NDLL_INTERP_LINEAR = 1,
  NDLL_INTERP_CUBIC = 2
};

/**
 * @brief Supported image formats
 */
enum NDLLImageType {
  NDLL_RGB = 0,
  NDLL_BGR = 1,
  NDLL_GRAY = 2
};

/**
 * @brief Supported tensor layouts
 */
enum NDLLTensorLayout {
  NDLL_NCHW = 0,
  NDLL_NHWC = 1
};

inline bool IsColor(NDLLImageType type) {
  if ((type == NDLL_RGB) || (type == NDLL_BGR)) {
    return true;
  }
  return false;
}

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

// Starts profiling NDLL
inline void NDLLProfilerStart() {
  cudaProfilerStart();
}

inline void NDLLProfilerStop() {
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
#ifdef NDLL_USE_NVTX
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
#ifdef NDLL_USE_NVTX
    if (started) {
      started = false;
      nvtxRangePop();
    }
#endif
  }

 private:
    bool started = false;
};

using std::to_string;

inline std::string to_string(const bool& b) {
  if (b) {
    return "True";
  } else {
    return "False";
  }
}

inline std::string to_string(const NDLLInterpType& interpolation) {
  switch (interpolation) {
    case NDLL_INTERP_NN:
      return "INTERP_NN";
    case NDLL_INTERP_LINEAR:
      return "INTERP_LINEAR";
    case NDLL_INTERP_CUBIC:
      return "INTERP_CUBIC";
    default:
      return "<unknown>";
  }
}

inline std::string to_string(const NDLLImageType& im_type) {
  switch (im_type) {
    case NDLL_RGB:
      return "RGB";
    case NDLL_BGR:
      return "BGR";
    case NDLL_GRAY:
      return "GRAY";
    default:
      return "<unknown>";
  }
}

inline std::string to_string(const NDLLTensorLayout& layout) {
  switch (layout) {
    case NDLL_NCHW:
      return "NCHW";
    case NDLL_NHWC:
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

}  // namespace ndll

#endif  // NDLL_COMMON_H_
