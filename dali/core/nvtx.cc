// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <unistd.h>
#include <pthread.h>
#include <memory>
#include "dali/core/nvtx.h"

namespace dali {

#if NVTX_ENABLED

class DomainTimeRangeImpl : RangeBase {
 public:
  static DomainTimeRangeImpl &GetInstance() {
    static DomainTimeRangeImpl impl;
    return impl;
  }

  void Start(const char *name, const uint32_t rgb) {
    nvtxEventAttributes_t att = {};
    FillAtrbs(att, name, rgb);
    nvtxDomainRangePushEx(dali_domain_, &att);
  }

  void Stop() {
    nvtxDomainRangePop(dali_domain_);
  }

 private:
  DomainTimeRangeImpl() {
    dali_domain_ = nvtxDomainCreateA("DALI");
  }

  ~DomainTimeRangeImpl() {
    nvtxDomainDestroy(dali_domain_);
  }

  nvtxDomainHandle_t dali_domain_;
};

DLL_PUBLIC DomainTimeRange::DomainTimeRange(const char *name, const uint32_t rgb) {
  DomainTimeRangeImpl::GetInstance().Start(name, rgb);
}

DLL_PUBLIC DomainTimeRange::~DomainTimeRange() {
  DomainTimeRangeImpl::GetInstance().Stop();
}

#endif  // NVTX_ENABLED

namespace {

void SetThreadNameInternal(const char *name) {
  char tmp_name[16];
  int i = 0;
  while (name[i] != '\0' && i < 15) {
    tmp_name[i] = name[i];
    ++i;
  }
  tmp_name[i] = '\0';
  pthread_setname_np(pthread_self(), tmp_name);
}

}  // namespace

DLL_PUBLIC void SetThreadName(const char *name) {
  #if NVTX_ENABLED
    nvtxNameOsThreadA(gettid(), name);
  #endif  // NVTX_ENABLED
  SetThreadNameInternal(name);
}


}  // namespace dali
