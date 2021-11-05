// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_DECODER_NVJPEG_FUSED_NVJPEG_DECODER_CROP_H_
#define DALI_OPERATORS_DECODER_NVJPEG_FUSED_NVJPEG_DECODER_CROP_H_

#include <vector>
#include "dali/operators/decoder/nvjpeg/nvjpeg_decoder_decoupled_api.h"
#include "dali/operators/image/crop/crop_attr.h"

namespace dali {

class nvJPEGDecoderCrop : public nvJPEGDecoder, protected CropAttr {
 public:
  explicit nvJPEGDecoderCrop(const OpSpec& spec)
    : nvJPEGDecoder(spec)
    , CropAttr(spec) {
  }

  DISABLE_COPY_MOVE_ASSIGN(nvJPEGDecoderCrop);

 protected:
  CropWindowGenerator GetCropWindowGenerator(int data_idx) const override {
    return CropAttr::GetCropWindowGenerator(data_idx);
  }

  void SetupSharedSampleParams(MixedWorkspace &ws) override {
    CropAttr::ProcessArguments(spec_, ws);
  }
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_NVJPEG_FUSED_NVJPEG_DECODER_CROP_H_
