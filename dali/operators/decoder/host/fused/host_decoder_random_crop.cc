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

#include <opencv2/opencv.hpp>
#include <memory>
#include <tuple>
#include <vector>
#include "dali/core/error_handling.h"
#include "dali/image/image_factory.h"
#include "dali/operators/decoder/host/fused/host_decoder_random_crop.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

DALI_REGISTER_OPERATOR(decoders__ImageRandomCrop, HostDecoderRandomCrop, CPU);
DALI_REGISTER_OPERATOR(ImageDecoderRandomCrop, HostDecoderRandomCrop, CPU);

}  // namespace dali
