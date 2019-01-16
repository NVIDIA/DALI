// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_SEQUENCE_SEQ_REARRANGE_H_
#define DALI_PIPELINE_OPERATORS_SEQUENCE_SEQ_REARRANGE_H_

#include <cstring>

#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class SequenceRearrange : public Operator<Backend> {
 public:
  inline explicit SequenceRearrange(const OpSpec &spec)
      : Operator<Backend>(spec), new_order_(spec.GetRepeatedArgument<int>("new_order")) {
    DALI_ENFORCE(new_order_.size() > 0, "Empty result sequence not allowed");
  }

  inline ~SequenceRearrange() override = default;

  DISABLE_COPY_MOVE_ASSIGN(SequenceRearrange);

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

 private:
  std::vector<int> new_order_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_SEQUENCE_SEQ_REARRANGE_H_
