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

#define MEL_FBANK_SUPPORTED_TYPES (float)

namespace dali {

static constexpr int kNumInputs = 1;
static constexpr int kNumOutputs = 1;

template <typename Backend>
class MelFilterBank : public Operator<Backend> {
 public:
  explicit MelFilterBank(const OpSpec &spec)
      : Operator<Backend>(spec) {
    args_.nfilter = spec.GetArgument<int>("nfilter");
    DALI_ENFORCE(args_.nfilter > 0, "number of filters should be > 0");

    args_.sample_rate = spec.GetArgument<float>("sample_rate");
    DALI_ENFORCE(args_.sample_rate > 0.0f, "sample rate should be > 0");

    args_.freq_low = spec.GetArgument<float>("freq_low");
    DALI_ENFORCE(args_.freq_low >= 0.0f, "freq_low should be >= 0");

    args_.freq_high = spec.GetArgument<float>("freq_high");
    if (args_.freq_high <= 0.0f)
      args_.freq_high = 0.5f * args_.sample_rate;
    DALI_ENFORCE(args_.freq_high > args_.freq_low && args_.freq_high <= args_.sample_rate,
      "freq_high should be within the range (freq_low, sample_rate/2]");

    args_.mel_formula = kernels::audio::MelScaleFormula::Slaney;
    if (spec.HasArgument("mel_formula")) {
      auto mel_formula = spec.GetArgument<std::string>("mel_formula");
      if (mel_formula == "htk") {
        args_.mel_formula = kernels::audio::MelScaleFormula::HTK;
      } else if (mel_formula == "slaney") {
        args_.mel_formula = kernels::audio::MelScaleFormula::Slaney;
      } else {
        DALI_FAIL(make_string("Unsupported mel_formula value \"", mel_formula,
          "\". Supported values are: \"slaney\", \"htk\""));
      }
    }

    args_.normalize = spec.GetArgument<bool>("normalize");
  }

 protected:
  bool CanInferOutputs() const override { return true; }
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;
  void RunImpl(workspace_t<Backend> &ws) override;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

  kernels::KernelManager kmgr_;
  kernels::KernelContext ctx_;
  kernels::audio::MelFilterBankArgs args_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_AUDIO_MEL_SCALE_MEL_FILTER_BANK_H_
