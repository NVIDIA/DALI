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

#ifndef DALI_PIPELINE_OPERATOR_SEQUENCE_INFO_H_
#define DALI_PIPELINE_OPERATOR_SEQUENCE_INFO_H_

#include <functional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "dali/core/common.h"

namespace dali {

struct FrameInfo {
  int sample_idx;
  int frame_idx = -1;
};

inline std::ostream &operator<<(std::ostream &os, const FrameInfo &frame_info) {
  os << "sample " << frame_info.sample_idx;
  if (frame_info.frame_idx >= 0) {
    os << ", frame " << frame_info.frame_idx;
  }
  return os;
}

// TODO(ktokarski) Consider:
// *  Making it a part of TensorList/TensorVector or TensorListShape
//    Pros: Available in most places where appliacble errors occur (not passed to kernels though)
//    How it would be handled in general, should it be removed
//    if someone, for instance, sets a layout, modifies shape of i-th sample?
// *  Making it a part of SourceInfo (DALIMeta)?
//    Pros: It is already there.
//    Cons: Should a sample know its index in a batch? (Maybe, it is called DALI*Meta* after all).
//    Also, not very optimistic: it means computing a string for every frame only to
//    discard it in successful case.
//  * Making it a part of SequenceWorkspaceView : workspace_t<Backend> - a wrapper around workspace
//    used inside SequenceOperators.
//    Pros: we already pass workspace to most of the places were the info is needed
//    (but not to kernels).
//    Cons: It requires guessing with RTTI/dynamic_cast if and which SequenceWorkspaceView<Backend>
//    we got. For example in utilities like GetGeneralizedArg, the static type is ArgumentWorkspace,
//    so no info about Backend and SequenceWorkspaceView.
// *  workspace/argument workspace:
//    Pros: we already pass workspace to most of the places were the info is needed
//    (but not to kernels).
//    Cons: Does frame mean anyting at that level? It is something that changes between
//    iterations.
//  * More generic error handling: maybe like DALI_FAIL(cond, [](Ctx ctx){make_string("Very
//    unexpected ", ctx.sample_info(sample_idx), " has been provided.")}) and catching
//    and re-throwing at the SequenceOp level.
//    Pros: not caring at all at the level of operator, kernel, utils about
//    passing anything sequence-op specific. Cons: Exceptions.
//
class SampleFrameCtx {
 public:
  using FrameInfoFn = std::function<FrameInfo(int)>;

  inline void AddArgumentInputInfoFn(const std::string &arg_name, FrameInfoFn fn) {
    DALI_ENFORCE(fn);
    arg_frame_info_[arg_name] = fn;
  }

  inline void AddInputInfoFn(FrameInfoFn fn) {
    DALI_ENFORCE(fn);
    input_frame_info_.push_back(fn);
  }

  inline const FrameInfoFn GetFrameInfo(int input_idx) const {
    if (0 <= input_idx && static_cast<size_t>(input_idx) < input_frame_info_.size()) {
      return input_frame_info_[input_idx];
    }
    return [](int sample_idx) -> FrameInfo { return {sample_idx}; };
  }

  inline const FrameInfoFn GetFrameInfo(const std::string &arg_name) const {
    auto it = arg_frame_info_.find(arg_name);
    if (it != arg_frame_info_.end()) {
      return it->second;
    }
    return [](int sample_idx) -> FrameInfo { return {sample_idx}; };
  }

  inline void Clear() {
    arg_frame_info_.clear();
    input_frame_info_.clear();
  }

 private:
  std::unordered_map<std::string, FrameInfoFn> arg_frame_info_;
  std::vector<FrameInfoFn> input_frame_info_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_SEQUENCE_INFO_H_
