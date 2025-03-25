// Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_PIPELINE_OUTPUT_DESC_H_
#define DALI_PIPELINE_PIPELINE_OUTPUT_DESC_H_

#include <string>
#include <vector>
#include "dali/pipeline/data/types.h"

namespace dali {

/**
 * Descriptor for an output, used by the Pipeline.
 *
 * Note: the Executor also has an output descriptor inside. It is different than this one.
 */
struct PipelineOutputDesc {
  std::string name;
  StorageDevice device = StorageDevice::CPU;
  DALIDataType dtype = DALI_NO_TYPE;
  int ndim = -1;
  TensorLayout layout;

  PipelineOutputDesc() = default;

  PipelineOutputDesc(
        std::string name,
        std::string_view device,
        DALIDataType dtype,
        int ndim,
        const TensorLayout &layout)
  : name(std::move(name))
  , device(ParseStorageDevice(device))
  , dtype(dtype)
  , ndim(ndim)
  , layout(layout) {}

  PipelineOutputDesc(const std::pair<std::string, std::string> &name_and_device)  // NOLINT
      : name(name_and_device.first),
        device(ParseStorageDevice(name_and_device.second)) {}

  bool operator==(const PipelineOutputDesc& other) const {
    return name == other.name && device == other.device && dtype == other.dtype &&
           ndim == other.ndim && layout == other.layout;
  }
};

inline std::ostream& operator<<(std::ostream& os, const PipelineOutputDesc& pod) {
  return os << "[Name: " << pod.name << "\tDevice: " << pod.device << "\tDtype: " << pod.dtype
            << "\tNdim: " << pod.ndim << "\tLayout: " << pod.layout << "]";
}

inline std::ostream& operator<<(std::ostream& os, const std::vector<PipelineOutputDesc>& pod) {
  for (size_t i = 0; i < pod.size(); i++) {
    os << "Output " << i << ": " << pod[i];
  }
  return os;
}

}  // namespace dali

#endif  // DALI_PIPELINE_PIPELINE_OUTPUT_DESC_H_
