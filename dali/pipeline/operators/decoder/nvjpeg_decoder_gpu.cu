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

#include "dali/pipeline/operators/decoder/nvjpeg_decoder_gpu.h"
#include "dali/pipeline/operators/op_schema.h"

namespace dali {

DALI_REGISTER_OPERATOR(nvJPEGDecoderGPUStage, nvJPEGDecoderGPUStage, Mixed);

DALI_SCHEMA(nvJPEGDecoderGPUStage)
  .DocStr(R"code(This operator is the GPU stage of nvJPEGDecoderNew, it is not supposed to be called separately.
It is automatically inserted during the pipeline creation.)code")
  .NumInput(3)
  .NumOutput(1)
  .MakeInternal()
  .AddParent("nvJPEGDecoderCPUStage");

}  // namespace dali


