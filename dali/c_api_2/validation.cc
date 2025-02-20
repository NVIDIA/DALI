// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime_api.h>
#include <stdexcept>
#include "dali/c_api_2/validation.h"

namespace dali::c_api {

void ValidateDeviceId(int device_id, bool allow_cpu_only) {
  if (device_id == CPU_ONLY_DEVICE_ID && allow_cpu_only)
    return;

  static int dev_count = []() {
    int ndevs = 0;
    CUDA_CALL(cudaGetDeviceCount(&ndevs));
    return ndevs;
  }();

  if (dev_count < 1)
    throw std::runtime_error("No CUDA device found.");

  if (device_id < 0 || device_id >= dev_count) {
    throw std::out_of_range(make_string(
        "The device id ", device_id, " is invalid."
        " Valid device ids are [0..", dev_count-1, "]."));
  }
}

}  // namespace dali::c_api
