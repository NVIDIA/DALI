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

#ifndef DALI_OPERATORS_SIGNAL_FFT_SPECTROGRAM_H_
#define DALI_OPERATORS_SIGNAL_FFT_SPECTROGRAM_H_

#include <memory>
#include <vector>
#include "dali/core/common.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

namespace detail {

template <typename Backend>
struct OpImplBase {
  virtual ~OpImplBase() = default;
  virtual bool SetupImpl(std::vector<OutputDesc> &output_desc,
                         const workspace_t<Backend> &ws) = 0;
  virtual void RunImpl(workspace_t<Backend> &ws) = 0;
};

}  // namespace detail

template <typename Backend>
class DLL_PUBLIC Spectrogram : public Operator<Backend> {
 public:
  DLL_PUBLIC Spectrogram(const OpSpec &spec);
  DLL_PUBLIC ~Spectrogram() override = default;

 protected:
  bool CanInferOutputs() const override { return true; }
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;
  void RunImpl(workspace_t<Backend> &ws) override;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

 private:
  OpSpec spec__;

  std::unique_ptr<detail::OpImplBase<Backend>> impl_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_SIGNAL_FFT_SPECTROGRAM_H_
