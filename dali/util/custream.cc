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

namespace dali {

CUStream::CUStream(int device_id, bool default_stream) : created_{false}, stream_{0} {
  if (!default_stream) {
    int orig_device;
    cudaGetDevice(&orig_device);
    auto set_device = false;
    if (device_id >= 0 && orig_device != device_id) {
      set_device = true;
      cudaSetDevice(device_id);
    }
    CUDA_CALL(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    created_ = true;
    if (set_device) {
      CUDA_CALL(cudaSetDevice(orig_device));
    }
  }
}


CUStream::~CUStream() {
  if (created_) {
    CUDA_CALL(cudaStreamDestroy(stream_));
  }
}


CUStream::CUStream(CUStream &&other) :
        created_{other.created_}, stream_{other.stream_} {
  other.stream_ = 0;
  other.created_ = false;
}


CUStream &CUStream::operator=(CUStream &&other) {
  stream_ = other.stream_;
  created_ = other.created_;
  other.stream_ = 0;
  other.created_ = false;
  return *this;
}


CUStream::operator cudaStream_t() {
  return stream_;
}

}  // namespace dali
