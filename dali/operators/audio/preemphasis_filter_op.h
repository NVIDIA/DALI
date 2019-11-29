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

#ifndef DALI_OPERATORS_AUDIO_PREEMPHASIS_FILTER_OP_H_
#define DALI_OPERATORS_AUDIO_PREEMPHASIS_FILTER_OP_H_

#include <random>
#include <vector>
#include "dali/core/convert.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {
namespace detail {

const std::string kCoeff = "preemph_coeff";  // NOLINT
const int kNumOutputs = 1;

}  // namespace detail

template<typename Backend>
class PreemphasisFilter : public Operator<Backend> {
 public:
  ~PreemphasisFilter() override = default;

  DISABLE_COPY_MOVE_ASSIGN(PreemphasisFilter);

 protected:
  explicit PreemphasisFilter(const OpSpec &spec) :
          Operator<Backend>(spec),
          output_type_(spec.GetArgument<std::remove_const_t<decltype(this->output_type_)>>(
                  arg_names::kDtype)) {}


  bool CanInferOutputs() const override {
    return true;
  }

  void AcquireArguments(const ArgumentWorkspace &ws) {
    this->GetPerSampleArgument(preemph_coeff_, detail::kCoeff, ws);
  }

  USE_OPERATOR_MEMBERS();
  std::vector<float> preemph_coeff_;
  const DALIDataType output_type_;
};


class PreemphasisFilterCpu : public PreemphasisFilter<CPUBackend> {
 public:
  explicit PreemphasisFilterCpu(const OpSpec &spec) : PreemphasisFilter(spec) {}

  ~PreemphasisFilterCpu() override = default;

  DISABLE_COPY_MOVE_ASSIGN(PreemphasisFilterCpu);

 protected:
  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const workspace_t<CPUBackend> &ws) override;

  void RunImpl(workspace_t<CPUBackend> &ws) override;
};


}  // namespace dali

#endif  // DALI_OPERATORS_AUDIO_PREEMPHASIS_FILTER_OP_H_
