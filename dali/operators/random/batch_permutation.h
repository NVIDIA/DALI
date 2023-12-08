// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_RANDOM_BATCH_PERMUTATION_H_
#define DALI_OPERATORS_RANDOM_BATCH_PERMUTATION_H_

#include <random>
#include <vector>
#include "dali/pipeline/operator/operator.h"
#include "dali/operators/random/rng_base_cpu.h"

namespace dali {

class BatchPermutation : public rng::OperatorWithRng<CPUBackend, false> {
 public:
  explicit BatchPermutation(const OpSpec &spec)
  : rng::OperatorWithRng<CPUBackend, false>(spec) {}

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    output_desc.resize(1);
    output_desc[0].shape = TensorListShape<0>(ws.GetRequestedBatchSize(0));
    output_desc[0].type = DALI_INT32;
    return true;
  }
  void RunImpl(Workspace &ws) override;
  bool CanInferOutputs() const override { return true; }
 private:
  void NoRepetitions(int N);
  void WithRepetitions(int N);
  vector<int> tmp_out_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_BATCH_PERMUTATION_H_
