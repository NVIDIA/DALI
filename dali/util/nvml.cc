// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/util/nvml.h"


namespace dali {
namespace nvml {
namespace impl {


float GetDriverVersion() {
  if (!nvmlIsInitialized()) {
    return 0;
  }

  float driver_version = 0;
  char version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];

  CUDA_CALL(nvmlSystemGetDriverVersion(version, sizeof version));
  driver_version = std::stof(version);
  return driver_version;
}


}  // namespace impl
}  // namespace nvml
}  // namespace dali
