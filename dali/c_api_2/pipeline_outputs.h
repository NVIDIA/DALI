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

#ifndef DALI_C_API_2_PIPELINE_OUTPUTS_H_
#define DALI_C_API_2_PIPELINE_OUTPUTS_H_

#include <string>
#include <vector>
#define DALI_ALLOW_NEW_C_API
#include "dali/dali.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/c_api_2/data_objects.h"

struct _DALIPipelineOutputs {
 protected:
  _DALIPipelineOutputs() = default;
  ~_DALIPipelineOutputs() = default;
};

namespace dali {
class Pipeline;
}  // namespace dali

namespace dali::c_api {

class PipelineOutputs : public _DALIPipelineOutputs {
 public:
  explicit PipelineOutputs(Pipeline *pipe, AccessOrder order = AccessOrder::host());

  RefCountedPtr<ITensorList> Get(int index);

  span<daliOperatorTrace_t> GetTraces();

  std::optional<std::string_view>
  GetTrace(std::string_view op_name, std::string_view trace_name) const;

 private:
  void ValidateOutputIdx(int idx) {
    if (idx < 0 || idx >= ws_.NumOutput()) {
      throw std::out_of_range(make_string("The output index ", idx, " is out of range. "
        "Valid range is [0..", ws_.NumOutput(), ")."));
    }
  }

  Workspace ws_;
  std::vector<RefCountedPtr<ITensorList>> output_wrappers_;
  // Use optional to implement lazy access with potentially empty result.
  std::optional<std::vector<daliOperatorTrace_t>> traces_;
};

PipelineOutputs *ToPointer(daliPipelineOutputs_h handle);

}  // namespace dali::c_api

#endif  // DALI_C_API_2_PIPELINE_OUTPUTS_H_
