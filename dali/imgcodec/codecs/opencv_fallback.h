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
#include "dali/imgcodec/codecs/codec_parallel_impl.h"

namespace dali {
namespace imgcodec {

/**
 * @brief A fallback codec, using OpenCV to decode the images
 */
class DLL_PUBLIC OpenCVCodecInstance : public BatchParallelCodecImpl<OpenCVCodecInstance> {
 public:
  using Base = BatchParallelCodecImpl<OpenCVCodecInstance>;
  OpenCVCodecInstance(int device_id, ThreadPool *tp) : Base(device_id, tp) {}

  using Base::Decode;

  DecodeResult Decode(SampleView<CPUBackend> out, EncodedImage *in, DecodeParams opts) override;

  DecodeResult Decode(SampleView<GPUBackend> out, EncodedImage *in, DecodeParams opts) override {
    throw std::logic_error("Backend not supported");
  }
};

class OpenCVCodec : public ImageCodec {
 public:
  ImageCodecProperties GetProperties() const override {
    static const auto props = [](){
      ImageCodecProperties props;
      props.supported_input_kinds =
        InputKind::HostMemory | InputKind::Filename;
      props.roi_support = false;
      props.fallback = false;
      return props;
    }();
    return props;
  }

  bool IsSupported(int device_id) const override {
    return device_id < 0;
  }

  private:
   std::shared_ptr<ImageCodecInstance> Create(int device_id, ThreadPool &tp) override{
    static auto instance = std::make_shared<OpenCVCodecInstance>(device_id, &tp);
    return instance;
   }
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_CODECS_OPENCV_FALLBACK_H_
