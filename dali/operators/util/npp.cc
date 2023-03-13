// Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/npp/npp.h"
#include "dali/core/error_handling.h"
#include "dali/core/cuda_error.h"
#include "dali/core/version_util.h"

namespace dali {

DLL_PUBLIC int NPPGetVersion() {
  auto version_s = nppGetLibVersion();
  int version = -1;
  if (version_s) {
    version = MakeVersionNumber(version_s->major, version_s->minor, version_s->build);
  }
  return version;
}

}  // namespace dali
