// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/support/random/coin_flip.h"

namespace ndll {

void CoinFlip::RunImpl(SupportWorkspace * ws, const int idx) {
  NDLL_ENFORCE(idx == 0, "CoinFlip does not support multiple input sets.");
  auto *output = ws->Output(idx);
  output->Resize({batch_size_});

  int * out_data = output->template mutable_data<int>();

  for (int i = 0; i < batch_size_; ++i) {
    out_data[i] = dis_(rng_) ? 1 : 0;
  }
}

NDLL_REGISTER_OPERATOR(CoinFlip, CoinFlip, Support);

NDLL_SCHEMA(CoinFlip)
  .DocStr("Produce tensor filled with 0s and 1s - results of random coin flip,"
      " usable as an argument for select ops.")
  .NumInput(0)
  .NumOutput(1)
  .AddOptionalArg("probability",
      R"code(`float`
      Probability of returning 1.)code", 0.5f);

}  // namespace ndll
