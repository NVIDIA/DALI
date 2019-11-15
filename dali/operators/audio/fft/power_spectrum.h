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

#ifndef DALI_OPERATORS_AUDIO_FFT_POWER_SPECTRUM_H_
#define DALI_OPERATORS_AUDIO_FFT_POWER_SPECTRUM_H_

#include <string>
#include <vector>
#include "dali/core/common.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/signal/fft/fft_cpu.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class PowerSpectrum : public Operator<Backend> {
 public:
  explicit inline PowerSpectrum(const OpSpec &spec)
      : Operator<Backend>(spec) {
    fft_args_.nfft = spec.GetArgument<int>("nfft");
    fft_args_.transform_axis = spec.GetArgument<int>("axis");
    auto spectrum_type_str = spec.GetArgument<std::string>("spectrum_type");
    if (spectrum_type_str == "magnitude") {
      fft_args_.spectrum_type = kernels::signal::fft::FFT_SPECTRUM_MAGNITUDE;
    } else if (spectrum_type_str == "power") {
      fft_args_.spectrum_type = kernels::signal::fft::FFT_SPECTRUM_POWER;
    } else {
      DALI_FAIL(make_string("Unexpected spectrum type: ", spectrum_type_str,
        ". Supported values are : power, magnitude"));
    }
  }

 protected:
  bool CanInferOutputs() const override { return true; }
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;
  void RunImpl(workspace_t<CPUBackend> &ws) override;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

  kernels::KernelManager kmgr_;
  kernels::signal::fft::FftArgs fft_args_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_AUDIO_FFT_POWER_SPECTRUM_H_
