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

#ifndef DALI_TEST_DUMMY_OP_H_
#define DALI_TEST_DUMMY_OP_H_

#include <vector>

#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class DummyOp : public Operator<Backend> {
 public:
  inline explicit DummyOp(const OpSpec &spec) :
    Operator<Backend>(spec) {}

  inline ~DummyOp() override = default;

  DISABLE_COPY_MOVE_ASSIGN(DummyOp);

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    return false;
  }

  void RunImpl(Workspace<Backend> &) override {
    DALI_FAIL("I'm a dummy op don't run me");
  }
};

}  // namespace dali

#endif  // DALI_TEST_DUMMY_OP_H_
