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

#ifndef DALI_TEST_OPERATORS_ORIGIN_TRACE_DUMP_H_
#define DALI_TEST_OPERATORS_ORIGIN_TRACE_DUMP_H_

#include <cstdint>
#include <string>
#include <vector>

#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/error_reporting.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/test/operators/string_msg_helper.h"

namespace dali {

class OriginTraceDump : public StringMsgHelper {
 public:
  inline explicit OriginTraceDump(const OpSpec &spec) : StringMsgHelper(spec) {}

  inline ~OriginTraceDump() override = default;

  DISABLE_COPY_MOVE_ASSIGN(OriginTraceDump);
  USE_OPERATOR_MEMBERS();

 protected:
  std::string GetMessage(const OpSpec &spec, const Workspace &ws) override {
    auto origin_stack_trace = GetOperatorOriginInfo(spec_);
    return FormatStack(origin_stack_trace, true);
  }
};

}  // namespace dali

#endif  // DALI_TEST_OPERATORS_ORIGIN_TRACE_DUMP_H_
