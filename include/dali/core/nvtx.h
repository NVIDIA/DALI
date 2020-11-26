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

#ifndef DALI_CORE_NVTX_H_
#define DALI_CORE_NVTX_H_

#include <cstdint>
#include <string>

#if NVTX_ENABLED
  // Just to get CUDART_VERSION value
  #include <cuda_runtime_api.h>
  #if (CUDART_VERSION >= 10000)
    #include "nvtx3/nvToolsExt.h"
  #elif (CUDART_VERSION < 10000) // NOLINT
    #include "nvToolsExt.h"
  #else
    #error Unknown CUDART_VERSION!
  #endif
#endif

#include "dali/core/api_helper.h"

namespace dali {


struct RangeBase {
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

  #if NVTX_ENABLED
  inline void FillAtrbs(nvtxEventAttributes_t &att, const char *name, const uint32_t rgb) {
    att.version = NVTX_VERSION;
    att.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    att.colorType = NVTX_COLOR_ARGB;
    att.color = rgb | 0xff000000;
    att.messageType = NVTX_MESSAGE_TYPE_ASCII;
    att.message.ascii = name;
  }
  #endif
};

// Basic timerange for profiling
struct TimeRange : RangeBase {
  explicit TimeRange(const std::string &name, const uint32_t rgb = kBlue)
    : TimeRange(name.c_str(), rgb) {}
  explicit TimeRange(const char *name, const uint32_t rgb = kBlue) {
  #if NVTX_ENABLED
    nvtxEventAttributes_t att = {};
    FillAtrbs(att, name, rgb);
    nvtxRangePushEx(&att);
    started = true;
  #endif
  }
  ~TimeRange() { stop(); }

  void stop() {
  #if NVTX_ENABLED
    if (started) {
      started = false;
      nvtxRangePop();
    }
  #endif
  }

 private:
  bool started = false;
};

struct DomainTimeRange : RangeBase {
  explicit DomainTimeRange(const std::string &name, const uint32_t rgb = kBlue)
    : DomainTimeRange(name.c_str(), rgb) {}
#if NVTX_ENABLED
  explicit DomainTimeRange(const char *name, const uint32_t rgb = kBlue);
  ~DomainTimeRange();
#else
  explicit DomainTimeRange(const char *name, const uint32_t rgb = kBlue) {}
  ~DomainTimeRange() {}
#endif
};

}  // namespace dali

#endif  // DALI_CORE_NVTX_H_
