// Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_DECODER_HOST_FUSED_HOST_DECODER_RANDOM_CROP_H_
#define DALI_OPERATORS_DECODER_HOST_FUSED_HOST_DECODER_RANDOM_CROP_H_

#include <string>
#include <vector>

#include "dali/core/common.h"
#include "dali/operators/image/crop/random_crop_attr.h"
#include "dali/operators/decoder/host/host_decoder.h"

namespace dali {

class RandomCropGenerator;

class HostDecoderRandomCrop : public OperatorWithRandomCrop<HostDecoder> {
 public:
  explicit HostDecoderRandomCrop(const OpSpec &spec)
    : OperatorWithRandomCrop<HostDecoder>(spec)
  {}

  inline ~HostDecoderRandomCrop() override = default;
  DISABLE_COPY_MOVE_ASSIGN(HostDecoderRandomCrop);

  using RandomCropAttr::GetCropWindowGenerator;
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_HOST_FUSED_HOST_DECODER_RANDOM_CROP_H_
