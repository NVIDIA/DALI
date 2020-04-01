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

#ifndef DALI_OPERATORS_GENERIC_ERASE_ERASE_H_
#define DALI_OPERATORS_GENERIC_ERASE_ERASE_H_

#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/tensor_shape.h"
#include "dali/kernels/scratch.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/operator_impl_utils.h"

namespace dali {

template <typename Backend>
class Erase : public Operator<Backend> {
 public:
  explicit inline Erase(const OpSpec &spec)
    : Operator<Backend>(spec) {}

 protected:
  using Operator<Backend>::RunImpl;
  void RunImpl(workspace_t<Backend> &ws) override;
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;

  bool CanInferOutputs() const override {
    return true;
  }

  USE_OPERATOR_MEMBERS();

  std::unique_ptr<OpImplBase<Backend>> impl_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_ERASE_ERASE_H_
