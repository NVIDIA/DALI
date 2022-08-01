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

#ifndef DALI_IMGCODEC_DECODERS_LIBJPEG_TURBO_H_
#define DALI_IMGCODEC_DECODERS_LIBJPEG_TURBO_H_

#include <memory>
#include <string>
#include "dali/imgcodec/image_decoder.h"
#include "dali/imgcodec/decoders/decoder_parallel_impl.h"

namespace dali {
namespace imgcodec {

/**
 * @brief JPEG decoder, using libjpeg-turbo.
 */
class DLL_PUBLIC LibJpegTurboDecoderInstance : public BatchParallelDecoderImpl {
 public:
  LibJpegTurboDecoderInstance(int device_id, ThreadPool *tp)
  : BatchParallelDecoderImpl(device_id, tp) {}

  using BatchParallelDecoderImpl::CanDecode;
  bool CanDecode(ImageSource *in, DecodeParams opts, const ROI &roi) override {
    return opts.format != DALI_YCbCr;  // not supported by libjpeg-turbo
  }

  using BatchParallelDecoderImpl::Decode;
  DecodeResult Decode(SampleView<CPUBackend> out,
                      ImageSource *in,
                      DecodeParams opts,
                      const ROI &roi) override;

  void SetParam(const char *name, const any &value) override {
    if (strcmp(name, "fast_idct") == 0) {
      use_fast_idct_ = any_cast<bool>(value);
    } else {
      DALI_FAIL("Unexpected param name: " + std::string(name));
    }
  }

  any GetParam(const char *name) const override {
    if (strcmp(name, "fast_idct") == 0) {
      return use_fast_idct_;
    } else {
      DALI_FAIL("Unexpected param name: " + std::string(name));
    }
  }

 private:
  bool use_fast_idct_ = false;
};

class LibJpegTurboDecoder : public ImageDecoder {
 public:
  ImageDecoderProperties GetProperties() const override {
    static const auto props = []() {
      ImageDecoderProperties props;
      props.supported_input_kinds = InputKind::HostMemory;
      props.supports_partial_decoding = true;
      props.fallback = true;
      return props;
    }();
    return props;
  }

  bool IsSupported(int device_id) const override {
    return device_id < 0;
  }

  std::shared_ptr<ImageDecoderInstance> Create(int device_id, ThreadPool &tp) const override {
    return std::make_shared<LibJpegTurboDecoderInstance>(device_id, &tp);
  }
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_LIBJPEG_TURBO_H_
