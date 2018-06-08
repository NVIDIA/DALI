// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_SUPPORT_RANDOM_UNIFORM_H_
#define NDLL_PIPELINE_OPERATORS_SUPPORT_RANDOM_UNIFORM_H_

#include <random>
#include <vector>

#include "ndll/pipeline/operators/operator.h"

namespace ndll {

class Uniform : public Operator<SupportBackend> {
 public:
  inline explicit Uniform(const OpSpec &spec) :
    Operator<SupportBackend>(spec),
    rng_(spec.GetArgument<int>("seed")) {
    std::vector<float> range = spec.GetRepeatedArgument<float>("range");
    NDLL_ENFORCE(range.size() == 2, "Range parameter needs to have 2 elements.");
    dis_ = std::uniform_real_distribution<float>(range[0], range[1]);
  }

  virtual inline ~Uniform() = default;

  DISABLE_COPY_MOVE_ASSIGN(Uniform);

  USE_OPERATOR_MEMBERS();

 protected:
  void RunImpl(Workspace<SupportBackend> * ws, const int idx) override;

 private:
  std::uniform_real_distribution<float> dis_;
  std::mt19937 rng_;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_SUPPORT_RANDOM_UNIFORM_H_
