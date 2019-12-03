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

#ifndef DALI_OPERATORS_AUDIO_MEL_SCALE_MEL_FILTER_BANK_H_
#define DALI_OPERATORS_AUDIO_MEL_SCALE_MEL_FILTER_BANK_H_

#include <cmath>
#include <string>
#include <vector>
#include "dali/core/common.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/audio/mel_scale/mel_filter_bank_args.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class MelFilterBank : public Operator<Backend> {
 public:
  explicit MelFilterBank(const OpSpec &spec)
      : Operator<Backend>(spec) {
    args_.nfilter = spec.GetArgument<int>("nfilter");
    DALI_ENFORCE(args_.nfilter > 0, "number of filters should be > 0");

    args_.sample_rate = spec.GetArgument<float>("sample_rate");
    DALI_ENFORCE(args_.sample_rate > 0.0f, "sample rate should be > 0");

    args_.fmin = spec.GetArgument<float>("fmin");
    DALI_ENFORCE(args_.fmin >= 0.0f, "fmin should be >= 0");

    args_.fmax = spec.GetArgument<float>("fmax");
    if (args_.fmax <= 0.0f)
      args_.fmax = 0.5f * args_.sample_rate;
    DALI_ENFORCE(args_.fmax > args_.fmin && args_.fmax <= args_.sample_rate,
      "fmax should be within the range (fmin, sample_rate/2]");
  }

 protected:
  bool CanInferOutputs() const override { return true; }
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;
  void RunImpl(workspace_t<CPUBackend> &ws) override;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

  kernels::KernelManager kmgr_;
  kernels::audio::MelFilterBankArgs args_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_AUDIO_MEL_SCALE_MEL_FILTER_BANK_H_
