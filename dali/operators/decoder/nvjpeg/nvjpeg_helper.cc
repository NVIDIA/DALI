// Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/decoder/nvjpeg/nvjpeg_helper.h"
#include "dali/core/version_util.h"

namespace dali {

DLL_PUBLIC int nvjpegGetVersion() {
  int major = -1;
  int minor = -1;
  int patch = -1;
  GetVersionProperty(nvjpegGetProperty, &major, MAJOR_VERSION, NVJPEG_STATUS_SUCCESS);
  GetVersionProperty(nvjpegGetProperty, &minor, MINOR_VERSION, NVJPEG_STATUS_SUCCESS);
  GetVersionProperty(nvjpegGetProperty, &patch, PATCH_LEVEL, NVJPEG_STATUS_SUCCESS);
  return MakeVersionNumber(major, minor, patch);
}

}  // namespace dali
