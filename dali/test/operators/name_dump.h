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

#ifndef DALI_TEST_OPERATORS_NAME_DUMP_H_
#define DALI_TEST_OPERATORS_NAME_DUMP_H_

#include <cstdint>
#include <string>
#include <vector>

#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/name_utils.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/test/operators/string_msg_helper.h"

namespace dali {

class NameDump : public StringMsgHelper {
 public:
  inline explicit NameDump(const OpSpec &spec) : StringMsgHelper(spec) {}

  inline ~NameDump() override = default;

  DISABLE_COPY_MOVE_ASSIGN(NameDump);
  USE_OPERATOR_MEMBERS();

 protected:
  std::string GetMessage(const OpSpec &spec, const Workspace &ws) override {
    auto target_function = spec.GetArgument<std::string>("target");
    auto include_module = spec.GetArgument<bool>("include_module");

    if (target_function == "module") {
      return GetOpModule(spec);
    } else if (target_function == "op_name") {
      return GetOpDisplayName(spec, include_module);
    }
    DALI_FAIL("Should not get here");
  }
};


}  // namespace dali

#endif  // DALI_TEST_OPERATORS_NAME_DUMP_H_
