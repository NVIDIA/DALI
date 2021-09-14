// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/dev_buffer.h"

#define MFCC_SUPPORTED_TYPES (float)

namespace dali {

namespace detail {

/**
 * @brief Liftering coefficients calculator
 */
template <typename Backend>
class LifterCoeffs {
  using Buffer = std::conditional_t<std::is_same<Backend, CPUBackend>::value,
                                    std::vector<float>, DeviceBuffer<float>>;

  void CalculateCoeffs(float *coeffs, int64_t offset, int64_t length) {
    float ampl_mult = lifter_ / 2;
    float phase_mult = static_cast<float>(M_PI) / lifter_;
    for (int64_t idx = 0, i = offset; idx < length; ++idx, ++i)
    coeffs[idx] = 1.f + ampl_mult * sin(phase_mult * (i + 1));
  }

 public:
  void Calculate(int64_t target_length, float lifter, cudaStream_t stream = 0);

  bool empty() const {
    return coeffs_.empty();
  }

  size_t size() const {
    return coeffs_.size();
  }

  const float* data() const {
    return coeffs_.data();
  }

 private:
  float lifter_ = 0;
  Buffer coeffs_;
};

}  // namespace detail


template <typename Backend>
class MFCC : public Operator<Backend> {
 public:
  using DctArgs = kernels::signal::dct::DctArgs;

  explicit MFCC(const OpSpec &spec)
      : Operator<Backend>(spec) {}

 protected:
  bool CanInferOutputs() const override { return true; }
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;
  void RunImpl(workspace_t<Backend> &ws) override;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

  void GetArguments(const workspace_t<Backend> &ws) {
    auto nsamples = ws.template InputRef<Backend>(0).shape().size();
    DctArgs arg;
    arg.ndct = spec_.template GetArgument<int>("n_mfcc");
    DALI_ENFORCE(arg.ndct > 0, "number of MFCCs should be > 0");

    arg.dct_type = spec_.template GetArgument<int>("dct_type");
    DALI_ENFORCE(arg.dct_type >= 1 && arg.dct_type <= 4,
      make_string("Unsupported DCT type: ", arg.dct_type, ". Supported types are: 1, 2, 3, 4."));

    arg.normalize = spec_.template GetArgument<bool>("normalize");
    if (arg.normalize) {
      DALI_ENFORCE(arg.dct_type != 1, "Ortho-normalization is not supported for DCT type I.");
    }

    axis_ = spec_.template GetArgument<int>("axis");
    DALI_ENFORCE(axis_ >= 0, "Provided axis cannot be negative.");

    lifter_ = spec_.template GetArgument<float>("lifter");
    args_.clear();
    args_.resize(nsamples, arg);
  }

  kernels::KernelManager kmgr_;
  kernels::KernelContext ctx_;
  std::vector<DctArgs> args_;
  int axis_;
  float lifter_ = 0.0f;
  detail::LifterCoeffs<Backend> lifter_coeffs_;
};



}  // namespace dali

#endif  // DALI_OPERATORS_AUDIO_MFCC_MFCC_H_
