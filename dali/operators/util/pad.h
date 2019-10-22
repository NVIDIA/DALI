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

#ifndef DALI_OPERATORS_UTIL_PAD_H_
#define DALI_OPERATORS_UTIL_PAD_H_

#include <cstring>
#include <utility>
#include <vector>

#include "dali/pipeline/operator/operator.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/common/pad.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/scratch.h"

namespace dali {

template <typename Backend>
class Pad : public Operator<Backend> {
 public:
  inline explicit Pad(const OpSpec &spec)
    : Operator<Backend>(spec),
    fill_value_(spec.GetArgument<float>("fill_value")),
    axes_(spec.GetRepeatedArgument<int>("axes")) {
    if (std::is_same<Backend, GPUBackend>::value) {
      kmgr_.Resize(1, 1);
    } else {
      kmgr_.Resize(num_threads_, batch_size_);
    }
  }

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;

  void RunImpl(Workspace<Backend> &ws) override;

  bool CanInferOutputs() const override {
    return true;
  }

  float fill_value_;
  std::vector<int> axes_;
  kernels::KernelManager kmgr_;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;
};

}  // namespace dali

#endif  // DALI_OPERATORS_UTIL_PAD_H_
