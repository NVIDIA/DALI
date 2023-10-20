// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_IMGCODEC_DECODERS_OPENCV_FALLBACK_H_
#define DALI_IMGCODEC_DECODERS_OPENCV_FALLBACK_H_

#include <map>
#include <memory>
#include <string>
#include "dali/imgcodec/image_decoder_interfaces.h"
#include "dali/imgcodec/decoders/decoder_parallel_impl.h"

namespace dali {
namespace imgcodec {

/**
 * @brief A fallback decoder, using OpenCV to decode the images
 */
class DLL_PUBLIC OpenCVDecoderInstance : public BatchParallelDecoderImpl {
 public:
  using Base = BatchParallelDecoderImpl;
  explicit OpenCVDecoderInstance(int device_id, const std::map<std::string, std::any> &params)
  : Base(device_id, params) {
    SetParams(params);
  }

  DecodeResult DecodeImplTask(int thread_idx,
                              SampleView<CPUBackend> out,
                              ImageSource *in,
                              DecodeParams opts,
                              const ROI &roi) override;
};

class OpenCVDecoderFactory : public ImageDecoderFactory {
 public:
  ImageDecoderProperties GetProperties() const override {
    static const auto props = []() {
      ImageDecoderProperties props;
      props.supported_input_kinds =
        InputKind::HostMemory | InputKind::Filename;
      props.supports_partial_decoding = false;  // roi support requires decoding the whole file
      props.fallback = false;  // this is the codec of last resort - if it fails, error out
      return props;
    }();
    return props;
  }

  bool IsSupported(int device_id) const override {
    return device_id < 0;
  }

  std::shared_ptr<ImageDecoderInstance> Create(
        int device_id, const std::map<std::string, std::any> &params = {}) const override {
    return std::make_shared<OpenCVDecoderInstance>(device_id, params);
  }
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_OPENCV_FALLBACK_H_
