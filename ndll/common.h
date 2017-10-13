#ifndef NDLL_COMMON_H_
#define NDLL_COMMON_H_

#include <cstdint>

#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cuda_fp16.h> // for __half & related methods
#include <cuda_profiler_api.h>

#ifdef NDLL_USE_NVTX
#include "nvToolsExt.h"
#endif

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


// Source: based on Caffe macro of the same name. Impl uses 'delete'
// instead of just making these functions private
//
// Helper to delete copy constructor & copy-assignment operator
#define DISABLE_COPY_MOVE_ASSIGN(name)          \
  name(const name&) = delete;                   \
  name& operator=(const name&) = delete;        \
  name(name&&) = delete;                        \
  name& operator=(name&&) = delete

// HACK: This global exists so that we have a way to enable/disable
// nvtx like we can with the cuda profiler. Could move this to a
// static variable in the TimeRange class
extern bool PROFILE;

// Starts profiling NDLL
inline void NDLLProfilerStart() {
  cudaProfilerStart();
  PROFILE = true;
}

inline void NDLLProfilerStop() {
  PROFILE = false;
  cudaProfilerStop();
}

// Basic timerange for profiling
struct TimeRange {
TimeRange(const char *name) {
#ifdef NDLL_USE_NVTX
  if (PROFILE) {
    nvtxRangePushA(name);
  }
#endif
}
~TimeRange() {
#ifdef NDLL_USE_NVTX
  if (PROFILE) {
    nvtxRangePop();
  }
#endif
}
};



} // namespace ndll

#endif // NDLL_COMMON_H_
