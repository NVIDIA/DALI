// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_OUTPUT_DESC_H
#define DALI_PIPELINE_OUTPUT_DESC_H
#include <string>
#include <vector>
#include "dali/pipeline/data/types.h"

namespace dali {
struct PipelineOutputDesc {
  std::string name, device;
  DALIDataType dtype;
  int ndim;

  PipelineOutputDesc() = default;

  PipelineOutputDesc(std::string name, std::string device, DALIDataType dtype, int ndim)
      : name(std::move(name)), device(std::move(device)), dtype(dtype), ndim(ndim) {}

  PipelineOutputDesc(const std::pair<std::string, std::string> &name_and_device)  // NOLINT
      : name(name_and_device.first),
        device(name_and_device.second),
        dtype(DALI_NO_TYPE),
        ndim(-1) {}
};
}  // namespace dali

#endif  // DALI_PIPELINE_OUTPUT_DESC_H
