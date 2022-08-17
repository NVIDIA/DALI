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

#ifndef DALI_IMGCODEC_IMAGE_DECODER_H_
#define DALI_IMGCODEC_IMAGE_DECODER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "dali/imgcodec/image_decoder_interfaces.h"
#include "dali/imgcodec/image_format.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/tensor_vector.h"

namespace dali {
namespace imgcodec {

class DLL_PUBLIC ImageDecoder : public ImageDecoderInstance, public ImageParser {
 public:
  explicit ImageDecoder(shared_ptr<ImageFormatRegistry> registry
                        = std::make_shared<ImageFormatRegistry>());

  bool CanParse(ImageSource *encoded) const override {
    return registry_->GetImageFormat(encoded) != nullptr;;
  }

  ImageInfo GetInfo(ImageSource *encoded) const override {
    if (auto *format = registry_->GetImageFormat(encoded))
      return format->Parser()->GetInfo(encoded);
    else
      DALI_FAIL(make_string("Cannot parse the image: ", encoded->SourceInfo()));
  }

  std::vector<bool> GetInfo(span<ImageInfo> info, span<ImageSource*> sources) {
    assert(info.size() == sources.size());
    std::vector<bool> ret(size(info), false);
    for (int i = 0, n = info.size(); i < n; i++) {
      if (auto *format = registry_->GetImageFormat(sources[i])) {
        info[i] = format->Parser()->GetInfo(sources[i]);
        ret[i] = true;
      } else {
        info[i] = {};
        ret[i] = false;
      }
    }
    return ret;
  }

  std::shared_ptr<ImageFormatRegistry> FormatRegistry() const {
    return registry_;
  }

  /**
   * @brief Stubbed; returns true.
   */
  bool CanDecode(ImageSource *in, DecodeParams opts, const ROI &roi = {}) override;

  /**
   * @brief Stubbed; returns true for all images in the batch.
   */
  std::vector<bool> CanDecode(cspan<ImageSource *> in,
                              DecodeParams opts,
                              cspan<ROI> rois = {}) override;

  DecodeResult Decode(SampleView<CPUBackend> out,
                      ImageSource *in,
                      DecodeParams opts,
                      const ROI &roi = {}) override;

  DecodeResult Decode(cudaStream_t stream,
                      SampleView<GPUBackend> out,
                      ImageSource *in,
                      DecodeParams opts,
                      const ROI &roi = {}) override;


  /**
   * @brief Decodes a single image to device buffers
   */
  std::vector<DecodeResult> Decode(span<SampleView<CPUBackend>> out,
                                   cspan<ImageSource *> in,
                                   DecodeParams opts,
                                   cspan<ROI> rois = {}) override;

  /**
   * @brief Decodes a single image to device buffers
   */
  std::vector<DecodeResult> Decode(TensorVector<CPUBackend> out,
                                   cspan<ImageSource *> in,
                                   DecodeParams opts,
                                   cspan<ROI> rois = {});

  /**
   * @brief Decodes a single image to device buffers
   */
  std::vector<DecodeResult> Decode(cudaStream_t stream,
                                   span<SampleView<GPUBackend>> out,
                                   cspan<ImageSource *> in,
                                   DecodeParams opts,
                                   cspan<ROI> rois = {}) override;

  /**
   * @brief Decodes a single image to device buffers
   */
  std::vector<DecodeResult> Decode(cudaStream_t stream,
                                   TensorList<GPUBackend> out,
                                   cspan<ImageSource *> in,
                                   DecodeParams opts,
                                   cspan<ROI> rois = {});

  /**
   * @brief Sets a value of a parameter.
   *
   * It sets a value of a parameter for all existing sub-decoders as well ones that will be
   * created in the future.
   *
   * This function succeeds even if no sub-decoder recognizes the key.
   * If the value is incorrect for one of the decoders, but that decoder has not yet
   * been constructed, an error may be thrown at a later time, when the decoder is instantiated.
   */
  bool SetParam(const char *key, const any &value) override;

  /**
   * @brief Gets a value previously passed to `SetParam` with the given key.
   */
  any GetParam(const char *key) const override;


 private:
  std::shared_ptr<ImageFormatRegistry> registry_;
  std::map<std::string, any> params_;
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_IMAGE_DECODER_H_
