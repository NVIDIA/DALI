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

#include "dali/c_api_2/pipeline.h"
#include "dali/c_api_2/error_handling.h"

namespace dali::c_api {

Pipeline *ToPointer(daliPipeline_h handle) {
  if (!handle)
    throw NullHandle("Pipeline");
  return static_cast<Pipeline *>(handle);
}

PipelineOutputs *ToPointer(daliPipelineOutputs_h handle) {
  if (!handle)
    throw NullHandle("PipelineOutputs");
  return static_cast<PipelineOutputs *>(handle);
}

}  // namespace dali::c_api

