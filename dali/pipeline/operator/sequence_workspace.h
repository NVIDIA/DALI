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
  using Base = workspace_t<Backend>;

  void AddArgumentInput(const std::string &arg_name, shared_ptr<TensorVector<CPUBackend>> input,
                        SampleFrameInfoFn fn) {
    Base::AddArgumentInput(arg_name, input);
    arg_frame_info_[arg_name] = fn;
  }

  void AddInput(typename Base::template input_t<CPUBackend> input, SampleFrameInfoFn fn) {
    Base::AddInput(input);
    input_frame_info_.push_back(fn);
  }

  void AddInput(typename Base::template input_t<GPUBackend> input, SampleFrameInfoFn fn) {
    Base::AddInput(input);
    input_frame_info_.push_back(fn);
  }

  void Clear() {
    Base::Clear();
    arg_frame_info_.clear();
    input_frame_info_.clear();
  }

  SampleFrameInfoFn GetFrameInfo(int input_idx) const {
    DALI_ENFORCE(0 <= input_idx && static_cast<size_t>(input_idx) < input_frame_info_.size());
    DALI_ENFORCE(input_frame_info_.size() == static_cast<size_t>(this->NumInput()));
    auto fn = input_frame_info_[input_idx];
    DALI_ENFORCE(fn);
    return fn;
  }

  SampleFrameInfoFn GetFrameInfo(const std::string &arg_name) const {
    auto it = arg_frame_info_.find(arg_name);
    DALI_ENFORCE(it != arg_frame_info_.end(), "Argument \"" + arg_name + "\" not found.");
    auto fn = it->second;
    DALI_ENFORCE(fn);
    return fn;
  }

 private:
  std::unordered_map<std::string, SampleFrameInfoFn> arg_frame_info_;
  std::vector<SampleFrameInfoFn> input_frame_info_;
};


// TODO(ktokarski) Consider making it a virtual method of workspace/argument workspace instead of
// resroting RTTI. For now it seems too sequence-operator-specific to place in workspace base class.
template <typename Backend>
const SampleFrameInfoFn get_frame_info(const workspace_t<Backend> &ws, int input_idx) {
  auto ws_ptr = dynamic_cast<const SequenceWorkspaceView<Backend> *>(&ws);
  if (!ws_ptr) {
    return [](int sample_idx) -> FrameInfo { return {sample_idx}; };
  }
  return ws_ptr->GetFrameInfo(input_idx);
}

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_SEQUENCE_WORKSPACE_H_
