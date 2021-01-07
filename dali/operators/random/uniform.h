// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_RANDOM_UNIFORM_H_
#define DALI_OPERATORS_RANDOM_UNIFORM_H_

#include <random>
#include <vector>

#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

class Uniform : public Operator<CPUBackend> {
 public:
  inline explicit Uniform(const OpSpec &spec) :
          Operator<CPUBackend>(spec),
          rng_(spec.GetArgument<int64_t>("seed")),
          discrete_mode_(spec.HasArgument("values")) {
    DALI_ENFORCE(!(spec.HasArgument("range") && spec.HasArgument("values")),
            "`range` and `set` arguments are mutually exclusive");

    if (discrete_mode_) {
      set_ = spec.GetRepeatedArgument<float>("values");
      DALI_ENFORCE(!set_.empty(), "`values` argument cannot be empty");
    } else {
      range_ = spec.GetRepeatedArgument<float>("range");
      DALI_ENFORCE(range_.size() == 2, "`range` argument shall contain precisely 2 values");
      DALI_ENFORCE(range_[0] < range_[1],
                   "Invalid range. It shall be left-closed [a, b), where a < b");
    }
  }

  inline ~Uniform() override = default;

  DISABLE_COPY_MOVE_ASSIGN(Uniform);

  USE_OPERATOR_MEMBERS();
  using Operator<CPUBackend>::RunImpl;

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const HostWorkspace &ws) override {
    output_desc.resize(1);
    auto curr_batch_size = ws.GetRequestedBatchSize(0);
    output_desc[0].type = TypeTable::GetTypeInfo(DALI_FLOAT);
    if (spec_.ArgumentDefined("shape")) {
      GetShapeArgument(output_desc[0].shape, spec_, "shape", ws, curr_batch_size);
    } else {
      output_desc[0].shape = uniform_list_shape(curr_batch_size, TensorShape<0>());
    }
    return true;
  }

  void RunImpl(HostWorkspace &ws) override;

 private:
  void AssignRange(HostWorkspace &ws);

  void AssignSet(HostWorkspace &ws);

  std::mt19937 rng_;
  const bool discrete_mode_;  // mode can't change throughout lifetime of this op, due to RNG
  std::vector<float> range_, set_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_UNIFORM_H_
