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

namespace dali {

class CUContext {
 public:
  CUContext();
  explicit CUContext(CUdevice device, unsigned int flags = 0);
  ~CUContext();

  // no copying
  CUContext(const CUContext&) = delete;
  CUContext& operator=(const CUContext&) = delete;

  CUContext(CUContext&& other);
  CUContext& operator=(CUContext&& other);

  operator CUcontext() const;

  void push() const;
  bool initialized() const;
 private:
  CUdevice device_;
  CUcontext context_;
  bool initialized_;
};

}  // namespace dali

#endif  // DALI_UTIL_CUCONTEXT_H_
