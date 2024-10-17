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

#ifndef DALI_TEST_OPERATORS_COPY_H_
#define DALI_TEST_OPERATORS_COPY_H_

#include <vector>

#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class CopyArgumentOp : public Operator<Backend> {
 public:
  inline explicit CopyArgumentOp(const OpSpec &spec) : Operator<Backend>(spec) {
    DALI_ENFORCE(spec.HasTensorArgument("to_copy"),
                 "This testing operator accepts only tensor argument inputs of type float.");
  }

  inline ~CopyArgumentOp() override = default;

  DISABLE_COPY_MOVE_ASSIGN(CopyArgumentOp);

 protected:
  bool HasContiguousOutputs() const override {
    return false;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return false;
  }

  void RunImpl(Workspace &ws) override {
    ws.Output<Backend>(0).Copy(ws.ArgumentInput("to_copy"), ws.stream());
  }
};

}  // namespace dali

#endif  // DALI_TEST_OPERATORS_COPY_H_
