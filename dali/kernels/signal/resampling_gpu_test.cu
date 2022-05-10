// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>
#include <numeric>
#include "dali/kernels/signal/resampling_gpu.h"
#include "dali/kernels/signal/resampling_test.h"


namespace dali {
namespace kernels {
namespace signal {
namespace resampling {

class ResamplingGPUTest : public ResamplingTest {
 public:
  void RunResampling(span<const float> in_rates, span<const float> out_rates,
                     bool downmix) override {
    ResamplerGPU<float> R;
    R.Initialize(16);

    KernelContext ctx;
    ctx.gpu.stream = 0;

    auto req = R.Setup(ctx, ttl_in_.gpu(), in_rates, out_rates, downmix);
    auto outref_sh = ttl_outref_.cpu().shape;
    auto in_batch_sh = ttl_in_.cpu().shape;
    for (int s = 0; s < outref_sh.size(); s++) {
      auto sh = req.output_shapes[0].tensor_shape_span(s);
      auto expected_sh = outref_sh.tensor_shape_span(s);
      auto in_sh = in_batch_sh.tensor_shape_span(s);
      ASSERT_EQ(sh.size(), in_sh.size());
      if (downmix) {
        ASSERT_EQ(sh[1], 1);
      } else {
        if (sh.size() > 1)
          ASSERT_EQ(sh[1], in_sh[1]);
      }
    }

    R.Run(ctx, ttl_out_.gpu(), ttl_in_.gpu(), in_rates, out_rates, downmix);

    CUDA_CALL(cudaStreamSynchronize(ctx.gpu.stream));
  }
};

TEST_F(ResamplingGPUTest, SingleChannel) {
  this->RunTest(8, 1, false);
}

TEST_F(ResamplingGPUTest, TwoChannel) {
  this->RunTest(3, 2, false);
}

TEST_F(ResamplingGPUTest, EightChannel) {
  this->RunTest(3, 8, false);
}

TEST_F(ResamplingGPUTest, ThreeChannelDownmix) {
  this->RunTest(3, 3, true);
}

}  // namespace resampling
}  // namespace signal
}  // namespace kernels
}  // namespace dali
