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

#include "dali/pipeline/operators/decoder/nvjpeg_decoder_cpu_crop.h"
#include <vector>

namespace dali {

DALI_REGISTER_OPERATOR(nvJPEGDecoderCPUStageCrop, nvJPEGDecoderCPUStageCrop, CPU);

DALI_SCHEMA(nvJPEGDecoderCPUStageCrop)
  .DocStr(
R"code(This operator is the CPU stage of nvJPEGDecoder with fused Slicing, it is not supposed to be called separately.
It is automatically inserted during the pipeline creation.
Partially decode JPEG images using the nvJPEG library and a cropping window.
Output of the decoder is on the GPU and uses `HWC` ordering.)code")
  .NumInput(1)
  .NumOutput(3)
  .MakeInternal()
  .AddParent("nvJPEGDecoderCPUStage")
  .AddParent("CropAttr");

}  // namespace dali
