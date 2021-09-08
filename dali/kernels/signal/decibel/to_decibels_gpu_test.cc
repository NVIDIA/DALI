// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <random>
#include <tuple>
#include <vector>
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "dali/kernels/signal/decibel/to_decibels_gpu.h"
#include "dali/kernels/signal/decibel/decibel_calculator.h"
#include "dali/kernels/kernel_manager.h"

namespace dali {
namespace kernels {
namespace signal {
namespace test {

using T = float;

using TestBase = ::testing::TestWithParam<
  std::tuple<std::vector<TensorShape<>>,  // data shape
             T /* mul */,
             T /* s_ref */,
             T /* min_ratio */,
             T /* data_max */,
             bool  /* ref_max */>>;

class ToDecibelsGpuTest : public TestBase {
 public:
  ToDecibelsGpuTest()
      : data_shape_(std::get<0>(GetParam())),
        mul_(std::get<1>(GetParam())),
        s_ref_(std::get<2>(GetParam())),
        min_ratio_(std::get<3>(GetParam())),
        data_max_(std::get<4>(GetParam())),
        ref_max_(std::get<5>(GetParam())) {
    in_.reshape(data_shape_);
  }

  ~ToDecibelsGpuTest() override = default;

 protected:
  void SetUp() final {
    std::mt19937 rng;
    UniformRandomFill(in_.cpu(), rng, 0.0, 1.0);
  }
  TensorListShape<> data_shape_;
  T mul_ = 10.0;
  T s_ref_ = 1.0;
  T min_ratio_ = 1e-8;
  T data_max_ = 1.0;
  bool ref_max_ = false;
  TestTensorList<T> in_;
};

TEST_P(ToDecibelsGpuTest, ToDecibelsGpuTest) {
  auto batch_size = data_shape_.num_samples();
  KernelContext ctx;
  ctx.gpu.stream = 0;

  ToDecibelsArgs<T> args;
  args.multiplier = mul_;
  args.s_ref = s_ref_;
  args.min_ratio = min_ratio_;
  args.ref_max = ref_max_;

  kernels::signal::ToDecibelsGpu<T> kernel;
  auto req = kernel.Setup(ctx, in_.gpu());

  ScratchpadAllocator scratch_alloc;
  scratch_alloc.Reserve(req.scratch_sizes);
  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  ASSERT_EQ(data_shape_, req.output_shapes[0]);
  TestTensorList<T> out;
  out.reshape(data_shape_);

  auto in_view_cpu = in_.cpu();

  std::vector<T> max_values(batch_size, 0.0);
  mm::uptr<T> max_values_gpu;
  InListGPU<T, 0> max_values_arg;
  if (args.ref_max) {
    for (int b = 0; b < batch_size; ++b) {
      int64_t sz = volume(data_shape_[b]);
      auto &max_val = max_values[b];
      const auto* sample_data = in_view_cpu.tensor_data(b);
      for (int idx = 0; idx < sz; idx++) {
        max_val = std::max(max_val, sample_data[idx]);
      }
    }
    max_values_gpu = mm::alloc_raw_unique<T, mm::memory_kind::device>(batch_size);
    CUDA_CALL(cudaMemcpy(max_values_gpu.get(), max_values.data(), batch_size * sizeof(T),
                         cudaMemcpyHostToDevice));
    max_values_arg = {max_values_gpu.get(),
                      TensorListShape<0>(batch_size)};
  }

  kernel.Run(ctx, out.gpu(), in_.gpu(), args, max_values_arg);
  CUDA_CALL(cudaStreamSynchronize(0));
  auto out_view_cpu = out.cpu();
  for (int b = 0; b < batch_size; ++b) {
    int64_t sz = volume(data_shape_[b]);
    auto *out_data = out_view_cpu.tensor_data(b);
    const auto *in_data = in_view_cpu.tensor_data(b);

    T s_ref = args.ref_max ? max_values[b] : args.s_ref;
    MagnitudeToDecibel<T> dB(args.multiplier, s_ref, args.min_ratio);
    for (int idx = 0; idx < sz; idx++) {
      ASSERT_NEAR(out_data[idx], dB(in_data[idx]), 1e-5) <<
        "Output data doesn't match in sample " << b << " reference (idx=" << idx << ")";
    }
  }
}

INSTANTIATE_TEST_SUITE_P(ToDecibelsGpuTest, ToDecibelsGpuTest, testing::Combine(
    testing::Values(std::vector<TensorShape<>>{TensorShape<>{10, 1}},
                    std::vector<TensorShape<>>{TensorShape<>{1, 10}}),  // shape
    testing::Values(10.0, 20.0),     // mul
    testing::Values(1.0, 1e-6),      // s_ref
    testing::Values(1e-8, 1e-20),    // min_ratio
    testing::Values(1.0, 10000.0),   // data_max
    testing::Values(true, false)));  // ref_max

}  // namespace test
}  // namespace signal
}  // namespace kernels
}  // namespace dali
