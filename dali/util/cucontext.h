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

#ifndef DALI_UTIL_CUCONTEXT_H_
#define DALI_UTIL_CUCONTEXT_H_

#include "dali/util/dynlink_cuda.h"
#include "dali/core/api_helper.h"

namespace dali {

class DLL_PUBLIC CUContext {
 public:
  DLL_PUBLIC CUContext();
  DLL_PUBLIC explicit CUContext(int device, unsigned int flags = 0);
  DLL_PUBLIC ~CUContext();

  // no copying
  DLL_PUBLIC CUContext(const CUContext&) = delete;
  DLL_PUBLIC CUContext& operator=(const CUContext&) = delete;

  DLL_PUBLIC CUContext(CUContext&& other);
  DLL_PUBLIC CUContext& operator=(CUContext&& other);

  DLL_PUBLIC operator CUcontext() const;

  DLL_PUBLIC bool push() const;
  DLL_PUBLIC void pop() const;
  DLL_PUBLIC bool initialized() const {
    return initialized_;
  }

 private:
  CUdevice device_;
  int device_id_;
  CUcontext context_;
  bool initialized_;
};

 /**
 * Simple RAII device handling:
 * Switch to new context on construction, back to old
 * context on destruction. Keeps increase ref count of
 * shared ptr to context to make sure it exists during destruction
 */
class DLL_PUBLIC ContextGuard {
 public:
  DLL_PUBLIC explicit ContextGuard(std::shared_ptr<CUContext> ctx) :
      revert_(false),
      cu_context_{ctx} {
    if (cu_context_->initialized()) {
      revert_ = cu_context_->push();
    }
  }

  DLL_PUBLIC ~ContextGuard() {
    // if cu_context_ was not initialized then revert_ is false anyway
    if (revert_) {
      cu_context_->pop();
    }
  }

 private:
  bool revert_;
  std::shared_ptr<CUContext> cu_context_;
};

}  // namespace dali

#endif  // DALI_UTIL_CUCONTEXT_H_
