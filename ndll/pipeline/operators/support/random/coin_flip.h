// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_SUPPORT_RANDOM_COIN_FLIP_H_
#define NDLL_PIPELINE_OPERATORS_SUPPORT_RANDOM_COIN_FLIP_H_

#include <random>

#include "ndll/pipeline/operators/operator.h"

namespace ndll {

class CoinFlip : public Operator<SupportBackend> {
 public:
  inline explicit CoinFlip(const OpSpec &spec) :
    Operator<SupportBackend>(spec),
    dis_(spec.GetArgument<float>("probability")),
    rng_(spec.GetArgument<int>("seed")),
    batch_size_(spec.GetArgument<int>("batch_size")) {}

  virtual inline ~CoinFlip() = default;

  DISABLE_COPY_MOVE_ASSIGN(CoinFlip);

 protected:
  void RunImpl(Workspace<SupportBackend> * ws, const int idx) override;

 private:
  std::bernoulli_distribution dis_;
  std::mt19937 rng_;
  int batch_size_;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_SUPPORT_RANDOM_COIN_FLIP_H_
