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
    rng_(spec.GetArgument<int64_t>("seed")) {
    std::vector<float> range;
    GetSingleOrRepeatedArg(spec, range, "range", 2);
    dis_ = std::uniform_real_distribution<float>(range[0], range[1]);

    std::vector<int> shape_arg{1};
    if (spec.HasArgument("shape"))
      shape_arg = spec.GetRepeatedArgument<int>("shape");
    shape_ = std::vector<int64_t>{std::begin(shape_arg), std::end(shape_arg)};
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
    output_desc[0].shape = uniform_list_shape(batch_size_, shape_);
    output_desc[0].type = TypeTable::GetTypeInfo(DALI_FLOAT);
    return true;
  }

  void RunImpl(HostWorkspace &ws) override;

 private:
  std::uniform_real_distribution<float> dis_;
  std::mt19937 rng_;
  TensorShape<> shape_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_UNIFORM_H_
