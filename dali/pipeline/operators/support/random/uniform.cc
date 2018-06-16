// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include <vector>

#include "dali/pipeline/operators/support/random/uniform.h"

namespace dali {

void Uniform::RunImpl(SupportWorkspace * ws, const int idx) {
  DALI_ENFORCE(idx == 0, "Uniform does not support multiple input sets.");
  auto *output = ws->Output(idx);
  output->Resize({batch_size_});

  float * out_data = output->template mutable_data<float>();

  for (int i = 0; i < batch_size_; ++i) {
    out_data[i] = dis_(rng_);
  }
}

DALI_REGISTER_OPERATOR(Uniform, Uniform, Support);

DALI_SCHEMA(Uniform)
  .DocStr("Produce tensor filled with uniformly distributed random numbers.")
  .NumInput(0)
  .NumOutput(1)
  .AddOptionalArg("range",
      R"code(`list of float`
      Range of produced random numbers)code", std::vector<float>({-1, 1}));

}  // namespace dali
