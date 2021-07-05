// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_COMMON_H_
#define DALI_CORE_COMMON_H_

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "dali/core/api_helper.h"
#include "dali/core/int_literals.h"

namespace dali {

// pi
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// arbitrary value for the CPU device ID
constexpr int CPU_ONLY_DEVICE_ID = -99999;

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
  COUNT = 3
};

// What device is this tensor stored on
enum class StorageDevice {
  CPU = 0,
  GPU = 1,
  COUNT = 2,
};

static std::string to_string(OpType op_type) {
  switch (op_type) {
    case OpType::CPU:
      return "cpu";
    case OpType::GPU:
      return "gpu";
    case OpType::MIXED:
      return "mixed";
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
  DALI_RGB          = 0,
  DALI_BGR          = 1,
  DALI_GRAY         = 2,
  DALI_YCbCr        = 3,
  DALI_ANY_DATA     = 4
};

inline bool IsColor(DALIImageType type) {
  return type == DALI_RGB || type == DALI_BGR || type == DALI_YCbCr;
}

inline int NumberOfChannels(DALIImageType type, int orig_nchannels = -1) {
  switch (type) {
    case DALI_GRAY:
      return 1;
    case DALI_RGB:
    case DALI_BGR:
    case DALI_YCbCr:
      return 3;
    case DALI_ANY_DATA:
    default:
      assert(orig_nchannels > 0);
      return orig_nchannels;
  }
}

// Helper to delete copy constructor & copy-assignment operator
#define DISABLE_COPY_MOVE_ASSIGN(name)   \
  name(const name&) = delete;            \
  name& operator=(const name&) = delete; \
  name(name&&) = delete;                 \
  name& operator=(name&&) = delete

// Helper to declare copy constructor & copy-assignment operator default
#define DEFAULT_COPY_MOVE_ASSIGN(name)    \
  name(const name&) = default;            \
  name& operator=(const name&) = default; \
  name(name&&) = default;                 \
  name& operator=(name&&) = default

// Util to declare anonymous variable
#define CONCAT_1(var1, var2) var1##var2
#define CONCAT_2(var1, var2) CONCAT_1(var1, var2)
#define ANONYMIZE_VARIABLE(name) CONCAT_2(name, __LINE__)

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
  for (const T &t : v) {
    ret += to_string(t);
    ret += ", ";
  }
  ret += "]";
  return ret;
}

DLL_PUBLIC std::vector<std::string> string_split(const std::string &s, const char delim);


}  // namespace dali

#ifdef DALI_VERBOSE_LOGS
  #define LOGS_ENABLED 1
#else
  #define LOGS_ENABLED 0
#endif

#define LOG_LINE \
  if (LOGS_ENABLED) \
  std::cout << __FILE__ << ":" << __LINE__ << ": "

#define ERROR_LOG \
  if (1) \
  std::cerr << __FILE__ << ":" << __LINE__ << ": "

static constexpr int kDaliDataloaderSeed = 524287;

#endif  // DALI_CORE_COMMON_H_
