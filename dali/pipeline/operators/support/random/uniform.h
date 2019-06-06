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

#ifndef DALI_PIPELINE_OPERATORS_SUPPORT_RANDOM_UNIFORM_H_
#define DALI_PIPELINE_OPERATORS_SUPPORT_RANDOM_UNIFORM_H_

#include <random>
#include <vector>

#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/common.h"

namespace dali {

class Uniform : public Operator<SupportBackend> {
 public:
  inline explicit Uniform(const OpSpec &spec) :
    Operator<SupportBackend>(spec),
    rng_(spec.GetArgument<int64_t>("seed")) {
    std::vector<float> range;
    GetSingleOrRepeatedArg(spec, range, "range", 2);
    dis_ = std::uniform_real_distribution<float>(range[0], range[1]);
  }

  inline ~Uniform() override = default;

  DISABLE_COPY_MOVE_ASSIGN(Uniform);

  USE_OPERATOR_MEMBERS();
  using Operator<SupportBackend>::RunImpl;

 protected:
  void RunImpl(Workspace<SupportBackend> * ws, const int idx) override;

 private:
  std::uniform_real_distribution<float> dis_;
  std::mt19937 rng_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_SUPPORT_RANDOM_UNIFORM_H_
