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

#ifndef DALI_C_API_2_PIPELINE_TEST_UTILS_H_
#define DALI_C_API_2_PIPELINE_TEST_UTILS_H_

#include "dali/c_api_2/test_utils.h"
#include "dali/c_api_2/pipeline.h"
#include "dali/c_api_2/managed_handle.h"

namespace dali::c_api::test {

inline PipelineHandle Deserialize(std::string_view s, const daliPipelineParams_t &params) {
  daliPipeline_h h = nullptr;
  CHECK_DALI(daliPipelineDeserialize(&h, s.data(), s.length(), &params));
  return PipelineHandle(h);
}

inline TensorListHandle GetOutput(daliPipelineOutputs_h h, int idx) {
  daliTensorList_h tl = nullptr;
  CHECK_DALI(daliPipelineOutputsGet(h, &tl, idx));
  return TensorListHandle(tl);
}

inline PipelineOutputsHandle PopOutputs(daliPipeline_h h) {
  daliPipelineOutputs_h raw_out_h;
  CHECK_DALI(daliPipelinePopOutputs(h, &raw_out_h));
  return PipelineOutputsHandle(raw_out_h);
}

}  // namespace dali::c_api::test

#endif  // DALI_C_API_2_PIPELINE_TEST_UTILS_H_
