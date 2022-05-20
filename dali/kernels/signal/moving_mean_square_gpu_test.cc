// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/signal/moving_mean_square_gpu.h"
#include <gtest/gtest.h>
#include <vector>
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/pipeline/data/views.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace kernels {
namespace signal {
namespace test {


template<class InputType>
class MovingMeanSquareGPU : public ::testing::Test {
 public:
  using In = int;
  using Out = float;

  TestTensorList<In> in_;
  TestTensorList<Out> out_;

  void SetUp() final {
    int nsamples = 4;

    TensorListShape<> sh = {{30, }, {1000, }, {40, }, {50, }};
    in_.reshape(sh);
    out_.reshape(sh);

    std::mt19937 rng;
    UniformRandomFill(in_.cpu(), rng, 0.0, 1.0);
  }

  void RunTest() {
    KernelContext ctx;
    ctx.gpu.stream = 0;
    DynamicScratchpad scratch;
    ctx.scratchpad = &scratch;

    int window_size = 8;
    MovingMeanSquareArgs args{window_size};

    auto out = out_.gpu().to_static<1>();
    auto in = in_.gpu().to_static<1>();
    MovingMeanSquareGpu<In> kernel;
    kernel.Run(ctx, out, in, args);

    int nsamples = in.size();
    for (int s = 0; s < nsamples; s++) {
      auto in = in_.cpu()[s].data;
      auto out = out_.cpu()[s].data;
      int64_t len = in_.cpu()[s].shape.num_elements();
      assert(len == out_.cpu()[s].shape.num_elements());

      auto mean_squared = [](int *start, int *pos, int window_size) {
        int *ptr = pos - window_size;
        if (ptr < start)
          ptr = start;
        float sum = 0;
        for (; ptr <= pos; ++ptr) {
          auto x = *ptr;
          sum += (x * x);
        }
        return sum / window_size;
      };

      for (int64_t i = 0; i < len; i++) {
        ASSERT_NEAR(mean_squared(&in[0], &in[i], window_size), out[i], 1e-5) << "Failed @ " << i;
      }
    }
  }
};

using TestTypes =
    ::testing::Types<int16_t, uint16_t, int8_t, uint8_t, int32_t, uint32_t, float, double>;
TYPED_TEST_SUITE(MovingMeanSquareGPU, TestTypes);

TYPED_TEST(MovingMeanSquareGPU, RunTest) {
  this->RunTest();
}

}  // namespace test
}  // namespace signal
}  // namespace kernels
}  // namespace dali
