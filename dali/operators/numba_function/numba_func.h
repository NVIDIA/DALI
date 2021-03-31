// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_NUMBA_FUNCTION_NUMBA_FUNC_H_
#define DALI_OPERATORS_NUMBA_FUNCTION_NUMBA_FUNC_H_

#include <vector>

#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class NumbaFunc : public Operator<Backend> {
 public:
  using Base = Operator<Backend>;
  using Workspace = workspace_t<Backend>;

  explicit NumbaFunc(const OpSpec &spec_);

 protected:
  bool CanInferOutputs() const override { return true; }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;

  void RunImpl(Workspace &ws) override;

 private:
  uint64_t fn_ptr_;
  uint64_t setup_fn_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_NUMBA_FUNCTION_NUMBA_FUNC_H_
