// Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_CORE_DEVICE_GUARD_H_
#define DALI_CORE_DEVICE_GUARD_H_

#include "dali/core/cuda_utils.h"
#include "dali/core/error_handling.h"

namespace dali {

// /**
//  * Simple RAII device handling:
//  * Switch to new device on construction, back to old
//  * device on destruction
//  */
class DLL_PUBLIC DeviceGuard {
 public:
  /// @brief Saves current device id and restores it upon object destruction
  DeviceGuard();

  /// @brief Saves current device id, sets a new one and switches back
  ///        to the original device upon object destruction.
  //         for device id < 0 it is no-op
  explicit DeviceGuard(int new_device);
  ~DeviceGuard();
 private:
  CUcontext old_context_;
};

}  // namespace dali

#endif  // DALI_CORE_DEVICE_GUARD_H_
