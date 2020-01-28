// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_AUDIO_NONSILENCE_OP_H_
#define DALI_OPERATORS_AUDIO_NONSILENCE_OP_H_

#include <utility>
#include <vector>
#include <dali/kernels/kernel_manager.h>
#include <gtest/gtest_prod.h>
#include <dali/pipeline/data/views.h>
#include <dali/kernels/signal/decibel/to_decibels_cpu.h>
#include "dali/core/convert.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/kernels/signal/moving_mean_square.h"
#include "dali/operators/audio/nonsilence_op_impl.h"

namespace dali {
namespace detail {

const std::string kCutoff = "cutoff_value";  // NOLINT
const int kNumOutputs = 2;
using OutputType = int;
static_assert(std::is_integral<OutputType>::value,
              "Operator return indices, thus OutputType shall be integral");




}  // namespace detail

template<typename Backend>
class NonsilenceOperator : public Operator<Backend> {
 public:
  ~NonsilenceOperator() override = default;

  DISABLE_COPY_MOVE_ASSIGN(NonsilenceOperator);

 protected:
  explicit NonsilenceOperator(const OpSpec &spec) :
          Operator<Backend>(spec),
          cutoff_(spec.GetArgument<float>(detail::kCutoff)) {}


  bool CanInferOutputs() const override {
    return true;
  }


  USE_OPERATOR_MEMBERS();
  const float cutoff_;
};

//class NonsilenceOperatorCpuImpl;

class NonsilenceOperatorCpu : public NonsilenceOperator<CPUBackend> {
 public:
  explicit NonsilenceOperatorCpu(const OpSpec &spec) : NonsilenceOperator<CPUBackend>(spec) {}


  ~NonsilenceOperatorCpu() override = default;

  DISABLE_COPY_MOVE_ASSIGN(NonsilenceOperatorCpu);

 protected:
  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,                 const workspace_t<CPUBackend> &ws) override;


  void RunImpl(workspace_t<CPUBackend> &ws) override;


 private:
};


}  // namespace dali

#endif  // DALI_OPERATORS_AUDIO_NONSILENCE_OP_H_
