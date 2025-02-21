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

#include <gtest/gtest.h>
#include <vector>
#include "dali/core/cuda_event.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/kernels/reduce/find_reduce.cuh"
#include "dali/pipeline/data/views.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace kernels {
namespace test {

template <typename T>
struct threshold {
  T value_;
  explicit threshold(T value = 0) : value_(value) {}

  DALI_HOST_DEV DALI_FORCEINLINE bool operator()(T x) const noexcept {
    return x >= value_;
  }
};

template <typename T>
void SequentialFillBothEnds(T *data, int64_t data_len, T start_value) {
  auto value = start_value;
  int i = 0;
  int j = data_len - 1;
  for (; i < j; value++) {
    data[i++] = value;
    data[j--] = value;
  }
  if (i == j)
    data[i] = value;
}

class FindReduceTestGPU : public ::testing::Test {
 public:
  using T = float;
  using Idx = int64_t;
  TestTensorList<T> in_;
  TestTensorList<Idx> out_first_;
  TestTensorList<Idx> out_last_;

  TestTensorList<Idx> ref_first_;
  TestTensorList<Idx> ref_last_;

  void SetUp() final {
    int nsamples = 5;

    // 1500 chosen so that it doesn't fit one CUDA block (32*32)
    TensorListShape<> sh = {{5, }, {10, }, {9, }, {7, }, {1500, }};
    TensorListShape<0> out_sh(nsamples);
    in_.reshape(sh);
    out_first_.reshape(out_sh);
    out_last_.reshape(out_sh);
    ref_first_.reshape(out_sh);
    ref_last_.reshape(out_sh);

    for (int i = 0; i < nsamples; i++) {
      auto v = in_.cpu()[i];
      SequentialFillBothEnds(v.data, static_cast<int64_t>(v.shape.num_elements()), T(0));
    }

    // Threshold = 3
    // Input 0:  0 1 2 1 0
    //         ^^
    // Input 1:  0 1 2 3 4 4 3 2 1 0
    //                 ^     ^
    // Input 2:  0 1 2 3 4 3 2 1 0
    //                 ^   ^
    // Input 3:  0 1 2 3 2 1 0
    //                 ^^
    // Input 4:  0 1 2 3 4 5 ... 5 4 3 2 1 0
    //                 ^             ^
    ref_first_.cpu()[0].data[0] = -1;
    ref_last_.cpu()[0].data[0] = -1;

    ref_first_.cpu()[1].data[0] = 3;
    ref_last_.cpu()[1].data[0] = 6;

    ref_first_.cpu()[2].data[0] = 3;
    ref_last_.cpu()[2].data[0] = 5;

    ref_first_.cpu()[3].data[0] = 3;
    ref_last_.cpu()[3].data[0] = 3;

    ref_first_.cpu()[4].data[0] = 3;
    ref_last_.cpu()[4].data[0] = 1496;
  }

  void RunTest() {
    KernelContext ctx;
    ctx.gpu.stream = 0;
    DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
    ctx.scratchpad = &dyn_scratchpad;

    auto out_first = out_first_.gpu().to_static<0>();
    auto out_last = out_last_.gpu().to_static<0>();
    auto in = in_.gpu().to_static<1>();
    int nsamples = in.size();

    using Predicate = threshold<float>;
    TensorListShape<0> scalar_sh(nsamples);
    TestTensorList<Predicate, 0> predicates;
    predicates.reshape(scalar_sh);
    for (int i = 0; i < nsamples; i++) {
      *(predicates.cpu()[i].data) = Predicate(3);
    }
    auto predicates_gpu = predicates.gpu();

    FindReduceGPU<Idx, T, Predicate, reductions::min> first;
    FindReduceGPU<Idx, T, Predicate, reductions::max> last;
    first.Setup(ctx, in.shape);
    last.Setup(ctx, in.shape);
    first.Run(ctx, out_first, in, predicates_gpu);
    last.Run(ctx, out_last, in, predicates_gpu);
    CUDA_CALL(cudaStreamSynchronize(ctx.gpu.stream));

    for (int s = 0; s < nsamples; s++) {
      auto begin = out_first_.cpu()[s].data[0];
      auto length = out_last_.cpu()[s].data[0];
      int64_t input_len = in_.cpu()[s].shape.num_elements();
      EXPECT_EQ(ref_first_.cpu()[s].data[0], begin);
      EXPECT_EQ(ref_last_.cpu()[s].data[0], length);
    }
  }

  void RunPerf(int nsamples = 64) {
    using T = float;
    using Idx = int64_t;
    int n_iters = 1000;

    KernelContext ctx;
    ctx.gpu.stream = 0;
    DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
    ctx.scratchpad = &dyn_scratchpad;

    TensorListShape<0> out_sh(nsamples);
    TensorListShape<> sh(nsamples, 1);
    for (int s = 0; s < nsamples; s++) {
      if (s % 4 == 0)
        sh.tensor_shape_span(s)[0] = 16000 * 60;
      else if (s % 4 == 1)
        sh.tensor_shape_span(s)[0] = 16000 * 120;
      else if (s % 4 == 2)
        sh.tensor_shape_span(s)[0] = 16000 * 30;
      else if (s % 4 == 3)
        sh.tensor_shape_span(s)[0] = 16000 * 90;
    }

    TestTensorList<T> in_data;
    in_data.reshape(sh);

    TestTensorList<Idx> out_first_, out_last_;
    out_first_.reshape(out_sh);
    out_last_.reshape(out_sh);

    std::mt19937 rng;
    UniformRandomFill(in_data.cpu(), rng, 0.0, 1.0);

    CUDAEvent start = CUDAEvent::CreateWithFlags(0);
    CUDAEvent end = CUDAEvent::CreateWithFlags(0);
    double total_time_ms = 0;
    int64_t in_elems = in_data.cpu().shape.num_elements();
    int64_t in_bytes = in_elems * sizeof(T);
    int64_t out_elems = nsamples;
    int64_t out_bytes = out_elems * sizeof(int64_t);
    std::cout << "FindReduce GPU Perf test.\n"
              << "Input contains " << in_elems << " elements.\n";

    auto out_first = out_first_.gpu().to_static<0>();
    auto in = in_data.gpu().to_static<1>();

    using Predicate = threshold<float>;
    TensorListShape<0> scalar_sh(nsamples);
    TestTensorList<Predicate, 0> predicates;
    predicates.reshape(scalar_sh);
    for (int i = 0; i < nsamples; i++) {
      *(predicates.cpu()[i].data) = Predicate(3);
    }
    auto predicates_gpu = predicates.gpu();
    FindReduceGPU<Idx, T, Predicate, reductions::min> first;
    for (int i = 0; i < n_iters; ++i) {
      CUDA_CALL(cudaDeviceSynchronize());

      DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
      ctx.scratchpad = &dyn_scratchpad;

      CUDA_CALL(cudaEventRecord(start));
      first.Setup(ctx, in.shape);
      first.Run(ctx, out_first, in, predicates_gpu);

      CUDA_CALL(cudaEventRecord(end));
      CUDA_CALL(cudaDeviceSynchronize());
      float time_ms;
      CUDA_CALL(cudaEventElapsedTime(&time_ms, start, end));
      total_time_ms += time_ms;
    }
    std::cout << "Bandwidth: " << n_iters * (in_bytes + out_bytes) / (total_time_ms * 1e6)
              << " GBs/sec" << std::endl;
  }
};

TEST_F(FindReduceTestGPU, RunTest) {
  this->RunTest();
}

TEST_F(FindReduceTestGPU, DISABLED_Benchmark) {
  this->RunPerf();
}

}  // namespace test
}  // namespace kernels
}  // namespace dali
