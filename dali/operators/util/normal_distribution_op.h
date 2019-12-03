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

const std::string kMean = "mean";      // NOLINT
const std::string kStddev = "stddev";  // NOLINT
const std::string kShape = "shape";    // NOLINT

const int kNumOutputs = 1;
auto kShapeDefaultValue = std::vector<int>{-1};

}  // namespace detail

template<typename Backend>
class NormalDistribution : public Operator<Backend> {
 public:
  ~NormalDistribution() override = default;

  DISABLE_COPY_MOVE_ASSIGN(NormalDistribution);

 protected:
  explicit NormalDistribution(const OpSpec &spec) :
          Operator<Backend>(spec),
          shape_(spec.GetArgument<std::remove_const_t<decltype(this->shape_)>>(detail::kShape)),
          seed_(spec.GetArgument<std::remove_const_t<decltype(this->seed_)>>("seed")),
          dtype_(spec.GetArgument<std::remove_const_t<decltype(this->dtype_)>>(
                  arg_names::kDtype)) {}


  bool CanInferOutputs() const override {
    return true;
  }


  void AcquireArguments(const workspace_t<CPUBackend> &ws) {
    this->GetPerSampleArgument(mean_, detail::kMean, ws);
    this->GetPerSampleArgument(stddev_, detail::kStddev, ws);
  }


  TensorListShape<> GetOutputShape(const workspace_t<CPUBackend> &ws) {
    DALI_ENFORCE(!(spec_.NumRegularInput() == 1 && IsShapeArgumentProvided(spec_)),
                 make_string("Incorrect operator invocation. "
                             "The operator cannot be called with both Input and `shape` argument"));
    if (spec_.NumRegularInput() == 1) {
      single_value_in_output_ = false;
      return ShapesFromInputTensorList(ws);
    } else if (IsShapeArgumentProvided(spec_)) {
      single_value_in_output_ = false;
      return ShapesFromArgument(ws);
    } else {
      single_value_in_output_ = true;
      return ShapeForDefaultConfig(ws);
    }
  }

  using Operator<Backend>::spec_;
  using Operator<Backend>::batch_size_;
  std::vector<float> mean_, stddev_;
  decltype(detail::kShapeDefaultValue) shape_;
  const int64_t seed_;
  const DALIDataType dtype_;

  /**
   * When this is true it means, that neither Input or Argument have been provided
   * and the operator should generate one scalar per tensor in a batch (aka `CoinFlip` mode)
   */
  bool single_value_in_output_ = false;

 private:
  TensorListShape<> ShapesFromInputTensorList(const workspace_t<CPUBackend> &ws) {
    return ws.template InputRef<CPUBackend>(0).shape();
  }


  TensorListShape<> ShapesFromArgument(const workspace_t<CPUBackend> &ws) {
    std::vector<int> shape;
    this->GetPerSampleArgument(shape, detail::kShape, ws);
    TensorListShape<> ret(batch_size_);
    for (int i = 0; i < batch_size_; i++) {
      ret.set_tensor_shape(i, shape);
    }
    return ret;
  }


  TensorListShape<> ShapeForDefaultConfig(const workspace_t<CPUBackend> &ws) {
    TensorListShape<> ret(batch_size_);
    for (int i = 0; i < batch_size_; i++) {
      ret.set_tensor_shape(i, {1});
    }
    return ret;
  }


  bool IsShapeArgumentProvided(const OpSpec &spec) {
    return spec_.HasArgument(detail::kShape);
  }
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
  static_assert(std::is_same<decltype(mean_), decltype(stddev_)>::value &&
                is_vector<decltype(mean_)>::value,
                "It's assumed, that both `mean` and `stddev` are vectors");
  using distribution_t = std::normal_distribution<decltype(mean_)::value_type>;
};


}  // namespace dali

#endif  // DALI_OPERATORS_UTIL_NORMAL_DISTRIBUTION_OP_H_
