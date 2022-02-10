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

#ifndef DALI_PIPELINE_OPERATOR_SEQUENCE_WORKSPACE_H_
#define DALI_PIPELINE_OPERATOR_SEQUENCE_WORKSPACE_H_

#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/sequence_info.h"
#include "dali/pipeline/util/backend2workspace_map.h"


namespace dali {

template <typename Backend>
class SequenceWorkspaceView : public workspace_t<Backend> {
 public:
  void SetFrameInfoFns(std::vector<SampleFrameInfoFn> &&input_sample_info) {
    input_sample_info_ = input_sample_info;
  }

  const SampleFrameInfoFn GetFrameInfoForInput(int input_idx) const {
    DALI_ENFORCE(0 <= input_idx && static_cast<size_t>(input_idx) < input_sample_info_.size());
    DALI_ENFORCE(input_sample_info_.size() == static_cast<size_t>(this->NumInput()));
    return input_sample_info_[input_idx];
  }

 private:
  std::vector<SampleFrameInfoFn> input_sample_info_;
};


// TODO(ktokarski) Consider making it a method of workspace/argument workspace instead of resroting
// RTTI. For now it seems too sequence-operator-specific to place in workspace base class.
template <typename Backend>
const SampleFrameInfoFn GetFrameInfoForInput(const workspace_t<Backend> &ws, int input_idx) {
  auto ws_ptr = dynamic_cast<const SequenceWorkspaceView<Backend> *>(&ws);
  if (!ws_ptr) {
    return {};
  }
  return ws_ptr->GetFrameInfoForInput(input_idx);
}

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_SEQUENCE_WORKSPACE_H_
