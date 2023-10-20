// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_TEST_OPERATORS_PASSTHROUGH_WITH_TRACE_H_
#define DALI_TEST_OPERATORS_PASSTHROUGH_WITH_TRACE_H_

#include <vector>

#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class PassthroughWithTraceOp : public Operator<Backend> {
 public:
  inline explicit PassthroughWithTraceOp(const OpSpec &spec) : Operator<Backend>(spec) {}

  inline ~PassthroughWithTraceOp() override = default;

  DISABLE_COPY_MOVE_ASSIGN(PassthroughWithTraceOp);

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return false;
  }


  void RunImpl(Workspace &ws) override {
    ws.Output<Backend>(0).ShareData(ws.Input<Backend>(0));
    ws.SetOperatorTrace("test_trace", make_string("test_value", iteration_id++));
  }


 private:
  size_t iteration_id = 0;
};

}  // namespace dali

#endif  // DALI_TEST_OPERATORS_PASSTHROUGH_WITH_TRACE_H_
