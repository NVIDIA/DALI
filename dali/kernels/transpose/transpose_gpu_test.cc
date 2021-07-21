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

#include "dali/kernels/transpose/transpose_gpu.h"  // NOLINT
#include <gtest/gtest.h>
#include <random>
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "dali/kernels/transpose/transpose_test.h"
#include "dali/kernels/scratch.h"
#include "dali/core/cuda_event.h"

namespace dali {
namespace kernels {


TEST(TransposeGPU, Test4DAll) {
  std::mt19937_64 rng;
  TensorListShape<> shape;
  int N = 20;
  int D = 4;

  TestTensorList<int> in, out, ref;

  TransposeGPU transpose;
  ScratchpadAllocator sa;

#ifdef NDEBUG
  int max_extent = 70;
#else
  int max_extent = 30;
#endif
  std::uniform_int_distribution<int> shape_dist(1, max_extent);
  std::uniform_int_distribution<int> small_shape_dist(1, 8);
  std::bernoulli_distribution small_last_dim;

  auto start = CUDAEvent::CreateWithFlags(0);
  auto end = CUDAEvent::CreateWithFlags(0);

  for (auto &perm : testing::Permutations4) {
    shape.resize(N, D);

    for (int i = 0; i < N; i++) {
      for (int d = 0; d < D; d++) {
        if (d == D - 1 && small_last_dim(rng))
          shape.tensor_shape_span(i)[d] = small_shape_dist(rng);
        else
          shape.tensor_shape_span(i)[d] = shape_dist(rng);
      }
    }
    std::cerr << "Testing permutation "
      << perm[0] << " " << perm[1] << " " << perm[2] << " " << perm[3] << "\ninput shape = \n"
      << shape << "\n";

    in.reshape(shape);
    auto in_cpu = in.cpu();
    UniformRandomFill(in_cpu, rng, 0, 1000);

    KernelContext ctx;
    auto req = transpose.Setup(ctx, shape, make_span(perm), sizeof(int));
    auto out_shape = req.output_shapes[0];
    ASSERT_EQ(out_shape.num_elements(), shape.num_elements());
    out.reshape(out_shape);
    ref.reshape(out_shape);

    sa.Reserve(req.scratch_sizes);
    auto scratch = sa.GetScratchpad();
    ctx.scratchpad = &scratch;

    auto in_gpu  = in.gpu();
    auto out_gpu = out.gpu();

    CUDA_CALL(cudaMemset(out_gpu.data[0], 0xff, shape.num_elements() * sizeof(int)));
    CUDA_CALL(cudaEventRecord(start, ctx.gpu.stream));
    transpose.Run<int>(ctx, out_gpu, in_gpu);
    CUDA_CALL(cudaEventRecord(end, ctx.gpu.stream));
    CUDA_CALL(cudaGetLastError());

    auto ref_cpu = ref.cpu();
    auto out_cpu = out.cpu();
    float time;
    CUDA_CALL(cudaEventElapsedTime(&time, start, end));
    time *= 1e+6;
    std::cerr << 2*shape.num_elements()*sizeof(int) / time << " GB/s" << "\n";

    for (int i = 0; i < N; i++) {
      testing::RefTranspose(ref_cpu.data[i], in_cpu.data[i],
                            in_cpu.tensor_shape_span(i).data(), perm, D);
    }

    Check(out_cpu, ref_cpu);
  }
}



template <typename T, typename RNG>
void RunPerfTest(RNG &rng, const TensorListShape<> &shape, span<const int> perm) {
  auto start = CUDAEvent::CreateWithFlags(0);
  auto end = CUDAEvent::CreateWithFlags(0);

  int N = shape.num_samples();
  int D = shape.sample_dim();

  TestTensorList<T> in, out, ref;
  TransposeGPU transpose;
  ScratchpadAllocator sa;

  in.reshape(shape);
  auto in_cpu = in.cpu();
  UniformRandomFill(in_cpu, rng, 0, 100);

  KernelContext ctx;
  auto req = transpose.Setup(ctx, shape, perm, sizeof(T));
  auto out_shape = req.output_shapes[0];
  ASSERT_EQ(out_shape.num_elements(), shape.num_elements());
  out.reshape(out_shape);
  ref.reshape(out_shape);

  sa.Reserve(req.scratch_sizes);
  auto scratch = sa.GetScratchpad();
  ctx.scratchpad = &scratch;

  auto in_gpu  = in.gpu();
  auto out_gpu = out.gpu();

  transpose.Run<T>(ctx, out_gpu, in_gpu);  // warm-up
  scratch = sa.GetScratchpad();
  CUDA_CALL(cudaMemset(out_gpu.data[0], 0xff, shape.num_elements() * sizeof(T)));
  CUDA_CALL(cudaEventRecord(start, ctx.gpu.stream));
  transpose.Run<T>(ctx, out_gpu, in_gpu);
  CUDA_CALL(cudaEventRecord(end, ctx.gpu.stream));
  CUDA_CALL(cudaGetLastError());

  auto ref_cpu = ref.cpu();
  auto out_cpu = out.cpu();
  float time;
  CUDA_CALL(cudaEventElapsedTime(&time, start, end));
  time *= 1e+6;
  std::cerr << 2*shape.num_elements()*sizeof(T) / time << " GB/s" << "\n";

  for (int i = 0; i < N; i++) {
    testing::RefTranspose(ref_cpu.data[i], in_cpu.data[i],
                          in_cpu.tensor_shape_span(i).data(), perm.data(), D);
  }

  Check(out_cpu, ref_cpu);
}


TEST(TransposeGPU, Perf2D) {
  std::mt19937_64 rng;
  TensorListShape<> shape;
  int N = 20;
  int D = 2;


  int min_extent = 32;
  int max_extent = 1000;
  std::uniform_int_distribution<int> shape_dist(min_extent, max_extent);

  int perm[] = { 1, 0 };
  shape.resize(N, D);

  for (int i = 0; i < N; i++)
      shape.set_tensor_shape(i, TensorShape<2>{ shape_dist(rng), shape_dist(rng) });

  std::cerr << "Permuting 4-byte data; permutation 1 0\ninput shape = \n" << shape << "\n";

  RunPerfTest<int>(rng, shape, make_span(perm));
}

TEST(TransposeGPU, PerfDeinterleave) {
  std::mt19937_64 rng;
  TensorListShape<> shape;
  int N = 10;
  int D = 3;

  int min_extent = 300;
  int max_extent = 1000;
  int channels = 3;
  std::uniform_int_distribution<int> shape_dist(min_extent, max_extent);

  int perm[] = { 2, 0, 1 };
  shape.resize(N, D);

  for (int i = 0; i < N; i++)
      shape.set_tensor_shape(i, TensorShape<3>{ shape_dist(rng), shape_dist(rng), channels });

  std::cerr << "Permuting 1-byte data; permutation 2 0 1\ninput shape = \n" << shape << "\n";

  RunPerfTest<uint8>(rng, shape, make_span(perm));
}


TEST(TransposeGPU, PerfInterleave) {
  std::mt19937_64 rng;
  TensorListShape<> shape;
  int N = 10;
  int D = 3;

  int min_extent = 300;
  int max_extent = 1000;
  int channels = 3;
  std::uniform_int_distribution<int> shape_dist(min_extent, max_extent);

  int perm[] = { 1, 2, 0 };
  shape.resize(N, D);

  for (int i = 0; i < N; i++)
      shape.set_tensor_shape(i, TensorShape<3>{ channels, shape_dist(rng), shape_dist(rng) });

  std::cerr << "Permuting 1-byte data; permutation 1 2 0\ninput shape = \n" << shape << "\n";

  RunPerfTest<uint8>(rng, shape, make_span(perm));
}


}  // namespace kernels
}  // namespace dali
