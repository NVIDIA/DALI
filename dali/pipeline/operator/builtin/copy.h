// Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_BUILTIN_COPY_H_
#define DALI_PIPELINE_OPERATOR_BUILTIN_COPY_H_

#include <cstring>
#include <vector>
#include <type_traits>

#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/kernels/common/scatter_gather.h"

namespace dali {

template <typename Backend>
class Copy : public StatelessOperator<Backend> {
 public:
  explicit Copy(const OpSpec &spec) :
    StatelessOperator<Backend>(spec) {}

  DISABLE_COPY_MOVE_ASSIGN(Copy);

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    output_desc.resize(1);
    output_desc[0].type = ws.GetInputDataType(0);
    output_desc[0].shape = ws.GetInputShape(0);
    return true;
  }

  void RunImpl(Workspace &ws) override;
};


}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_COPY_H_
