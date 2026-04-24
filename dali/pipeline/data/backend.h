// Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_DATA_BACKEND_H_
#define DALI_PIPELINE_DATA_BACKEND_H_

#include <cuda_runtime_api.h>
#include <memory>
#include <string_view>

#include "dali/core/error_handling.h"
#include "dali/core/cuda_error.h"

namespace dali {

class OpSpec;

class DLL_PUBLIC CPUBackend {};
class DLL_PUBLIC GPUBackend {};
class DLL_PUBLIC MixedBackend {};

template <typename Backend>
constexpr std::string_view BackendDeviceName = "<unsupported>";

template <>
constexpr std::string_view BackendDeviceName<CPUBackend> = "cpu";
template <>
constexpr std::string_view BackendDeviceName<GPUBackend> = "gpu";
template <>
constexpr std::string_view BackendDeviceName<MixedBackend> = "mixed";

// This is defined only for backends that map to actual storage devices
template <typename Backend>
struct backend_to_storage_device;

template <>
struct backend_to_storage_device<CPUBackend>
    : std::integral_constant<StorageDevice, StorageDevice::CPU> {};

template <>
struct backend_to_storage_device<GPUBackend>
    : std::integral_constant<StorageDevice, StorageDevice::GPU> {};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_BACKEND_H_
