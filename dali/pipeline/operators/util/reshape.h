// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_UTIL_RESHAPE_H_
#define DALI_PIPELINE_OPERATORS_UTIL_RESHAPE_H_

#include <cstring>

#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class Reshape : public Operator<Backend> {
 public:
  inline explicit Reshape(const OpSpec &spec)
      : Operator<Backend>(spec), new_shape_(spec.GetRepeatedArgument<Index>("new_shape")) {
    DALI_ENFORCE(!new_shape_.empty(), "New shape cannot be empty");
    if (new_shape_.size() == 1 && new_shape_[0] == kUseInputForShapes) {
      use_input_for_shapes_ = true;
    }
  }

  inline ~Reshape() override = default;

  DISABLE_COPY_MOVE_ASSIGN(Reshape);

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

 private:
  std::vector<Index> new_shape_;
  bool use_input_for_shapes_{false};

  static constexpr int kInTensorIdx = 0;
  static constexpr int kInShapeIdx = 1;
  static constexpr int kOutTensorIdx = 0;
  static constexpr int kUseInputForShapes = -1;

  std::vector<Index> GetNewShapeForSample(SampleWorkspace *ws);
  std::vector<std::vector<Index>> GetNewShapesForSamples(DeviceWorkspace *ws);
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_UTIL_RESHAPE_H_
