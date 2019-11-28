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
const std::string kShape = "shape";    // NOLINT
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
          seed_(spec.GetArgument<std::remove_const_t<decltype(this->seed_)>>("seed")),
          dtype_(spec.GetArgument<std::remove_const_t<decltype(this->dtype_)>>(detail::kDtype)) {}


  bool CanInferOutputs() const override {
    return true;
  }


  void AcquireArguments(const workspace_t<CPUBackend> &ws) {
    this->GetPerSampleArgument(mean_, detail::kMean, ws);
    this->GetPerSampleArgument(stddev_, detail::kStddev, ws);
  }


  TensorListShape<> GetOutputShape(const workspace_t<CPUBackend> &ws) {
    if (spec_.NumRegularInput() == 1) {
      single_value_in_output_ = false;
      return ws.template InputRef<CPUBackend>(0).shape();
    } else if (spec_.NumArgumentInput() == 1) {
      single_value_in_output_ = false;
      std::vector<int> shape;
      this->GetPerSampleArgument(shape, detail::kShape, ws);
      TensorListShape<> ret(batch_size_);
      for (int i = 0; i < batch_size_; i++) {
        ret.set_tensor_shape(i, shape);
      }
      return ret;
    } else if (spec_.NumRegularInput() == 0 && spec_.NumArgumentInput() == 0) {
      single_value_in_output_ = true;
      TensorListShape<> ret(batch_size_);
      for (int i = 0; i < batch_size_; i++) {
        ret.set_tensor_shape(i, {1});
      }
      return ret;
    } else {
      DALI_FAIL(make_string(
              "Operator called with wrong arguments. This operator can be called with: one Input, "
              "one ArgumentInput or no inputs. Detected: [", spec_.NumRegularInput(),
              " Inputs] and [", spec_.NumArgumentInput(), " ArgumentInputs]."));
    }
  }


  USE_OPERATOR_MEMBERS();
  std::vector<float> mean_, stddev_;
  const int64_t seed_;
  const DALIDataType dtype_;

  /// When this is true it means, that neither Input or ArgumentInput have been provided
  bool single_value_in_output_ = false;
};


class NormalDistributionCpu : public NormalDistribution<CPUBackend> {
 public:
  explicit NormalDistributionCpu(const OpSpec &spec) : NormalDistribution(spec), rng_(seed_) {}

  ~NormalDistributionCpu() override = default;

  DISABLE_COPY_MOVE_ASSIGN(NormalDistributionCpu);

 protected:
  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const workspace_t<CPUBackend> &ws) override;

  void RunImpl(workspace_t<CPUBackend> &ws) override;

 private:
  void AssignTensorToOutput(workspace_t<CPUBackend> &ws);

  void AssignSingleValueToOutput(workspace_t<CPUBackend> &ws);

  std::mt19937_64 rng_;
  static_assert(std::is_same<decltype(mean_), decltype(stddev_)>::value, "");
  static_assert(is_vector<decltype(mean_)>::value,
                "It's assumed, that both `mean` and `stddev` are vectors (for ArgumentInput)");
  using distribution_t = std::normal_distribution<decltype(mean_)::value_type>;
};


}  // namespace dali

#endif  // DALI_OPERATORS_UTIL_NORMAL_DISTRIBUTION_OP_H_
