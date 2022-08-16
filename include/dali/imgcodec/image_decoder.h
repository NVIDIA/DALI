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

#include <memory>
#include <map>
#include "dali/imgcodec/image_decoder_interfaces.h"
#include "dali/imgcodec/image_format.h"


namespace dali {
namespace imgcodec {

class DLL_PUBLIC ImageDecoder {
 public:
  explicit ImageDecoder(shared_ptr<ImageFormatRegistry> registry
                        = std::make_shared<ImageFormatRegistry>());

  ImageInfo GetInfo(ImageSource *encoded) const {
  }

  std::vector<bool> GetInfo(span<ImageInfo> info, span<ImageSource*> sources) {
  }

  std::shared_ptr<ImageFormatRegistry> FormatRegistry() const {
    return registry_;
  }

  /**
   * @brief Decodes a single image to a host buffer
   */
  DecodeResult Decode(SampleView<CPUBackend> out,
                      ImageSource *in,
                      DecodeParams opts,
                      const ROI &roi = {}) = 0;

  /**
   * @brief Decodes a single image to device buffers
   */
  std::vector<DecodeResult> Decode(cudaStream_t stream,
                                   span<SampleView<GPUBackend>> out,
                                   cspan<ImageSource *> in,
                                   DecodeParams opts,
                                   cspan<ROI> rois = {}) = 0;

  /**
   * @brief Decodes a single image to device buffers
   */
  std::vector<DecodeResult> Decode(cudaStream_t stream,
                                   TensorListView<GPUBackend>> out,
                                   cspan<ImageSource *> in,
                                   DecodeParams opts,
                                   cspan<ROI> rois = {}) = 0;


 private:
  std::shared_ptr<ImageFormatRegistry> registry_;
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_IMAGE_DECODER_H_
