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

#ifndef DALI_OPERATORS_AUDIO_MFCC_MFCC_H_
#define DALI_OPERATORS_AUDIO_MFCC_MFCC_H_

#include <cmath>
#include <string>
#include <vector>
#include "dali/core/common.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/signal/dct/dct_args.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class MFCC : public Operator<Backend> {
 public:
  explicit MFCC(const OpSpec &spec)
      : Operator<Backend>(spec) {
    args_.ndct = spec.GetArgument<int>("n_mfcc");
    DALI_ENFORCE(args_.ndct > 0, "number of MFCCs should be > 0");

    args_.dct_type = spec.GetArgument<int>("dct_type");
    DALI_ENFORCE(args_.dct_type >= 1 && args_.dct_type <= 4,
      make_string("Unsupported DCT type: ", args_.dct_type, ". Supported types are: 1, 2, 3, 4."));

    args_.normalize = spec.GetArgument<bool>("normalize");
    if (args_.normalize) {
      DALI_ENFORCE(args_.dct_type != 1, "Ortho-normalization is not supported for DCT type I");
    }

    args_.axis = spec.GetArgument<int>("axis");
    DALI_ENFORCE(args_.axis >= 0);

    lifter_ = spec.GetArgument<float>("lifter");
  }

 protected:
  bool CanInferOutputs() const override { return true; }
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;
  void RunImpl(workspace_t<CPUBackend> &ws) override;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

  kernels::KernelManager kmgr_;
  kernels::signal::dct::DctArgs args_;
  float lifter_ = 0.0f;
};

}  // namespace dali

#endif  // DALI_OPERATORS_AUDIO_MFCC_MFCC_H_
