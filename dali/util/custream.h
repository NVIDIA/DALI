// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_UTIL_CUSTREAM_H_
#define DALI_UTIL_CUSTREAM_H_

#include <driver_types.h>
#include "dali/common.h"

namespace dali {

class DLL_PUBLIC CUStream {
 public:
  CUStream(int device_id, bool default_stream, int priority);

  ~CUStream();

  CUStream(const CUStream &) = delete;

  CUStream &operator=(const CUStream &) = delete;

  explicit CUStream(CUStream &&);

  CUStream &operator=(CUStream &&);

  operator cudaStream_t();

 private:
  cudaStream_t stream_;
};

}  // namespace dali

#endif  // DALI_UTIL_CUSTREAM_H_
