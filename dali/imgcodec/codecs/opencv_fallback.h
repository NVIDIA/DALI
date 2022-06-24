// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_IMGCODEC_CODECS_OPENCV_FALLBACK_H_
#define DALI_IMGCODEC_CODECS_OPENCV_FALLBACK_H_

#include <memory>
#include "dali/imgcodec/image_codec.h"
#include "dali/imgcodec/codecs/parallel_impl.h"

namespace dali {
namespace imgcodec {

class OpenCVCodecInstance : public BatchParallelCodecImpl<OpenCVCodecInstance> {
 public:
  using Base = BatchParallelCodecImpl<OpenCVCodecInstance>;
  OpenCVCodecInstance(int device_id, ThreadPool *tp) : Base(device_id, tp) {}
};

class OpenCVCodec : public ImageCodec {
  public:
  private:
   static std::shared_ptr<OpenCVCodecInstance> Create(int, ThreadPool &) {
    static auto instance = std::make_shared<OpenCVCodecInstance>();
    return instance;
   }
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_CODECS_OPENCV_FALLBACK_H_
