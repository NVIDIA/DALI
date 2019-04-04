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

#include "dali/util/custream.h"
#include "dali/util/cuda_utils.h"
#include "dali/util/device_guard.h"

namespace dali {

CUStream::CUStream(int device_id, bool default_stream, int priority) :
        stream_{0} {
  if (!default_stream) {
    DeviceGuard dg(device_id);
    CUDA_CALL(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, priority));
  }
}


CUStream::~CUStream() {
  if (stream_ != 0) {
    auto err = cudaStreamDestroy(stream_);
    if (err != cudaSuccess) {
      std::cerr << "Critical error in destroying stream: " << err << std::endl;
      std::terminate();
    }
  }
}


CUStream::CUStream(CUStream &&other) :
        stream_{other.stream_} {
  other.stream_ = 0;
}


CUStream &CUStream::operator=(CUStream &&other) {
  std::swap(stream_, other.stream_);
  other.~CUStream();
  return *this;
}


CUStream::operator cudaStream_t() {
  return stream_;
}

}  // namespace dali
