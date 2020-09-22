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

#ifndef DALI_OPERATORS_SIGNAL_DECIBEL_TO_DECIBELS_OP_H_
#define DALI_OPERATORS_SIGNAL_DECIBEL_TO_DECIBELS_OP_H_

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include "dali/core/common.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/signal/decibel/to_decibels_args.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/operator_impl_utils.h"

static constexpr int kNumInputs = 1;
static constexpr int kNumOutputs = 1;

namespace dali {

template <typename Backend>
class ToDecibels : public Operator<Backend> {
 public:
  explicit ToDecibels(const OpSpec &spec)
      : Operator<Backend>(spec) {
    args_.multiplier = spec.GetArgument<float>("multiplier");
    args_.ref_max = !spec.HasArgument("reference");
    if (!args_.ref_max) {
      args_.s_ref = spec.GetArgument<float>("reference");
      DALI_ENFORCE(args_.s_ref != 0, "`reference` argument can't be zero");
    }
    auto cutoff_db = spec.GetArgument<float>("cutoff_db");
    args_.min_ratio = std::pow(10.0f, cutoff_db / args_.multiplier);
    if (args_.min_ratio == 0)
      args_.min_ratio = std::nextafter(0.0f, 1.0f);
  }

 protected:
  bool CanInferOutputs() const override { return true; }
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;
  void RunImpl(workspace_t<Backend> &ws) override;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

  kernels::KernelManager kmgr_;
  kernels::signal::ToDecibelsArgs<float> args_;

  std::unique_ptr<OpImplBase<Backend>> impl_;
  DALIDataType type_ = DALI_NO_TYPE;
};

}  // namespace dali

#endif  // DALI_OPERATORS_SIGNAL_DECIBEL_TO_DECIBELS_OP_H_
