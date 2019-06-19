// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include <string>
#include "dali/pipeline/operators/decoder/nvjpeg/legacy_api/nvjpeg_decoder.h"

namespace dali {

DALI_SCHEMA(nvJPEGDecoder)
  .DocStr(R"code(Specific implementation of `ImageDecoder` `mixed` backend.
**Deprecated** Use `ImageDecoder` instead
)code")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ImageDecoder")

DALI_REGISTER_OPERATOR(nvJPEGDecoder, nvJPEGDecoder, Mixed);
DALI_REGISTER_OPERATOR(ImageDecoder, nvJPEGDecoder, Mixed);

}  // namespace dali
