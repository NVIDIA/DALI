// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <benchmark/benchmark.h>
#include <vector>
#include "dali/benchmark/dali_bench.h"
#include "dali/kernels/slice/slice_gpu.cuh"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "dali/kernels/dynamic_scratchpad.h"

namespace dali {

constexpr int Dims = 3;
using InputType = float;
using OutputType = float;

class SliceBenchGPU : public DALIBenchmark {
 public:
  kernels::TestTensorList<InputType, Dims> test_data;
  kernels::TestTensorList<OutputType, Dims> out_data;

  void Setup(const TensorShape<Dims> &in_shape,
             const TensorShape<Dims> &out_shape,
             int batch_size = 1) {
    test_data.reshape(uniform_list_shape<Dims>(batch_size, in_shape));
    InputType num = 0;
    auto seq_gen = [&num]() { return num++; };
    Fill(test_data.cpu(), seq_gen);
    out_data.reshape(uniform_list_shape<Dims>(batch_size, out_shape));
  }

  void RunGPU(benchmark::State& st) {
    int H = st.range(0);
    int W = st.range(1);
    int C = st.range(2);
    int anchor_h = st.range(3);
    int anchor_w = st.range(4);
    int anchor_c = st.range(5);
    int crop_h = st.range(6);
    int crop_w = st.range(7);
    int crop_c = st.range(8);
    int batch_size = st.range(9);

    TensorShape<Dims> in_shape{H, W, C};
    TensorShape<Dims> anchor{anchor_h, anchor_w, anchor_c};
    TensorShape<Dims> out_shape{crop_h, crop_w, crop_c};
    Setup(in_shape, out_shape, batch_size);

    using Kernel = kernels::SliceGPU<OutputType, InputType, Dims>;
    Kernel kernel;

    std::vector<kernels::SliceArgs<OutputType, Dims>> args_vec(batch_size);
    for (auto &args : args_vec) {
      args.anchor = anchor;
      args.shape = out_shape;
    }

    auto out_tv = out_data.gpu();
    auto in_tv = test_data.gpu();

    for (auto _ : st) {
      kernels::KernelContext ctx;
      ctx.gpu.stream = 0;

      auto req = kernel.Setup(ctx, in_tv, args_vec);

      kernels::DynamicScratchpad dyn_scratchpad(ctx.gpu.stream);
      ctx.scratchpad = &dyn_scratchpad;

      kernel.Run(ctx, out_tv, in_tv, args_vec);
      CUDA_CALL(cudaStreamSynchronize(ctx.gpu.stream));
      st.counters["FPS"] = benchmark::Counter(st.iterations() + 1,
        benchmark::Counter::kIsRate);
    }
  }
};

static void SliceKernelArgs_GPU_OnlySlice(benchmark::internal::Benchmark *b) {
  for (int H = 1000; H >= 500; H /= 2) {
    int W = H, C = 3;
    int crop_h = 9 * H / 10;
    int crop_w = 9 * W / 10;
    b->Args({H, W, C, 0, 0, 0, crop_h, crop_w, C, 1});
    b->Args({H, W, C, 0, 0, 0, crop_h, crop_w, C, 10});
  }
}

BENCHMARK_DEFINE_F(SliceBenchGPU, Slice_GPU_OnlySlice)(benchmark::State& st) {
  this->RunGPU(st);
}

BENCHMARK_REGISTER_F(SliceBenchGPU, Slice_GPU_OnlySlice)->Iterations(1000)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->Apply(SliceKernelArgs_GPU_OnlySlice);

static void SliceKernelArgs_GPU_OnlyPad(benchmark::internal::Benchmark *b) {
  for (int H = 1000; H >= 500; H /= 2) {
    int W = H, C = 3;
    int crop_h = 9 * H / 10;
    int crop_w = 9 * W / 10;
    int anchor_h = H;
    int anchor_w = W;
    b->Args({H, W, C, anchor_h, anchor_w, 0, crop_h, crop_w, C, 1});
    b->Args({H, W, C, anchor_h, anchor_w, 0, crop_h, crop_w, C, 10});
  }
}

BENCHMARK_DEFINE_F(SliceBenchGPU, Slice_GPU_OnlyPad)(benchmark::State& st) {
  this->RunGPU(st);
}

BENCHMARK_REGISTER_F(SliceBenchGPU, Slice_GPU_OnlyPad)->Iterations(1000)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->Apply(SliceKernelArgs_GPU_OnlyPad);

static void SliceKernelArgs_GPU_SliceAndPad(benchmark::internal::Benchmark *b) {
  for (int H = 1000; H >= 500; H /= 2) {
    int W = H, C = 3;
    int crop_h = 9 * H / 10;
    int crop_w = 9 * W / 10;
    int anchor_h = H / 2;
    int anchor_w = W / 2;
    b->Args({H, W, C, anchor_h, anchor_w, 0, crop_h, crop_w, C, 1});
    b->Args({H, W, C, anchor_h, anchor_w, 0, crop_h, crop_w, C, 10});
  }
}

BENCHMARK_DEFINE_F(SliceBenchGPU, Slice_GPU_SliceAndPad)(benchmark::State& st) {
  this->RunGPU(st);
}

BENCHMARK_REGISTER_F(SliceBenchGPU, Slice_GPU_SliceAndPad)->Iterations(1000)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->Apply(SliceKernelArgs_GPU_SliceAndPad);


}  // namespace dali
