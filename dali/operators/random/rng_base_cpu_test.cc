// Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <string>

#include "dali/operators/random/rng_base_cpu.h"
#include "dali/pipeline/pipeline.h"

namespace dali {

class RNGCheckpointingTest : public ::testing::Test {
 protected:
  template<class DataType>
  void RunOperatorTest(const std::string &name) {
    constexpr int batch_size = 16;
    constexpr int iterations = 10;

    Pipeline original_pipe(batch_size, 1, 0);
    Pipeline restored_pipe(batch_size, 1, 0);
    for (Pipeline *pipe : {&original_pipe, &restored_pipe}) {
      pipe->AddOperator(
        OpSpec(name)
        .AddOutput("data_out", StorageDevice::CPU), "rng_op");
      std::vector<std::pair<string, string>> outputs = {{"data_out", "cpu"}};
      pipe->Build(outputs);
    }

    Workspace ws;
    auto run_iteration = [&](Pipeline &pipe) {
      pipe.Run();
      pipe.Outputs(&ws);

      std::vector<DataType> result;
      auto shape = ws.Output<CPUBackend>(0).AsTensor().shape();
      for (int nr = 0; nr < shape[0]; nr++)
        result.push_back(ws.Output<CPUBackend>(0).tensor<DataType>(0)[nr]);

      return result;
    };

    // make sure that the results differ run to run
    EXPECT_NE(run_iteration(original_pipe), run_iteration(original_pipe));

    // warmup, mutates the internal operator state
    for (int i = 0; i < iterations; i++)
      run_iteration(original_pipe);

    // save and restore the pipeline
    auto op = original_pipe.GetOperator("rng_op");
    OpCheckpoint cpt("rng_op");
    op->SaveState(cpt, {});
    restored_pipe.GetOperator("rng_op")->RestoreState(cpt);

    // make sure the restored pipeline has the same internal state
    for (int i = 0; i < iterations; i++)
      EXPECT_EQ(run_iteration(original_pipe), run_iteration(restored_pipe));
  }
};

TEST_F(RNGCheckpointingTest, CoinFlip) {
  RunOperatorTest<int32_t>("random__CoinFlip");
}

TEST_F(RNGCheckpointingTest, Uniform) {
  RunOperatorTest<float>("random__Uniform");
}

TEST_F(RNGCheckpointingTest, Normal) {
  RunOperatorTest<float>("random__Normal");
}

}  // namespace dali
