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
TimeRange(std::string name) {
#ifdef NDLL_USE_NVTX
  nvtxRangePushA(name.c_str());
#endif
}
~TimeRange() {
#ifdef NDLL_USE_NVTX
  nvtxRangePop();
#endif
}
};

using std::to_string;

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

}  // namespace ndll

#endif  // NDLL_COMMON_H_
