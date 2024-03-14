// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_TEST_OPERATORS_DEPRECATION_WARNING_H_
#define DALI_TEST_OPERATORS_DEPRECATION_WARNING_H_

#include <cstdint>
#include <string>
#include <vector>

#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/name_utils.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/test/operators/string_msg_helper.h"

namespace dali {

class DeprecationWarningOp : public Operator<CPUBackend> {
 public:
  inline explicit DeprecationWarningOp(const OpSpec &spec) : Operator<CPUBackend>(spec) {}

  inline ~DeprecationWarningOp() override = default;

  DISABLE_COPY_MOVE_ASSIGN(DeprecationWarningOp);
  USE_OPERATOR_MEMBERS();

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override  {
    return false;
  }

  void RunImpl(Workspace &ws) override {}
};


}  // namespace dali

#endif  // DALI_TEST_OPERATORS_DEPRECATION_WARNING_H_
