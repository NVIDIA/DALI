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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_HOST_DECODER_CROP_H_
#define DALI_PIPELINE_OPERATORS_DECODER_HOST_DECODER_CROP_H_

#include <vector>
#include "dali/common.h"
#include "dali/pipeline/operators/decoder/host_decoder.h"
#include "dali/pipeline/operators/crop/crop_attr.h"

namespace dali {

class HostDecoderCrop : public HostDecoder, protected CropAttr {
 public:
  explicit HostDecoderCrop(const OpSpec &spec);

  inline ~HostDecoderCrop() override = default;
  DISABLE_COPY_MOVE_ASSIGN(HostDecoderCrop);

  void inline SetupSharedSampleParams(SampleWorkspace *ws) override {
    CropAttr::ProcessArguments(ws);
  }

 protected:
  inline CropWindowGenerator GetCropWindowGenerator(int data_idx) const override {
    return CropAttr::GetCropWindowGenerator(data_idx);
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_HOST_DECODER_CROP_H_
