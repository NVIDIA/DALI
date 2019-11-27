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

#ifndef DALI_OPERATORS_UTIL_NORMAL_DISTRIBUTION_OP_H_
#define DALI_OPERATORS_UTIL_NORMAL_DISTRIBUTION_OP_H_

#include <random>
#include <vector>
#include "dali/core/convert.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {
namespace detail {

/**
 * Names of arguments
 */
const std::string kMean = "mean";      // NOLINT
const std::string kStddev = "stddev";  // NOLINT
const std::string kDtype = "dtype";    // NOLINT
const int kNumOutputs = 1;

}  // namespace detail

template<typename Backend>
class NormalDistribution : public Operator<Backend> {
 public:
  ~NormalDistribution() override = default;

  DISABLE_COPY_MOVE_ASSIGN(NormalDistribution);

 protected:
  explicit NormalDistribution(const OpSpec &spec) :
          Operator<Backend>(spec),
          mean_(spec.GetArgument<std::remove_const_t<decltype(this->mean_)>>(detail::kMean)),
          stddev_(spec.GetArgument<std::remove_const_t<decltype(this->stddev_)>>(detail::kStddev)),
          seed_(spec.GetArgument<std::remove_const_t<decltype(this->seed_)>>("seed")),
          dtype_(spec.GetArgument<std::remove_const_t<decltype(this->dtype_)>>(detail::kDtype)) {}


  bool CanInferOutputs() const override {
    return true;
  }


  USE_OPERATOR_MEMBERS();
  const float mean_, stddev_;
  const int64_t seed_;
  const DALIDataType dtype_;
};


class NormalDistributionCpu : public NormalDistribution<CPUBackend> {
 public:
  explicit NormalDistributionCpu(const OpSpec &spec) : NormalDistribution(spec), rng_(),
                                                       distribution_(mean_, stddev_) {}


  ~NormalDistributionCpu() override = default;

  DISABLE_COPY_MOVE_ASSIGN(NormalDistributionCpu);

 protected:
  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const workspace_t<CPUBackend> &ws) override;

  void RunImpl(workspace_t<CPUBackend> &ws) override;

  std::mt19937_64 rng_;
  static_assert(std::is_same<decltype(mean_), decltype(stddev_)>::value, "");
  std::normal_distribution<std::remove_const_t<decltype(mean_)>> distribution_;
};


}  // namespace dali

#endif  // DALI_OPERATORS_UTIL_NORMAL_DISTRIBUTION_OP_H_
