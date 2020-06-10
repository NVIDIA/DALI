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
      << perm[0] << " " << perm[1] << " " << perm[2] << " " << perm[3] << "  input shape "
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
    transpose.Run<int>(ctx, out_gpu, in_gpu);
    CUDA_CALL(cudaGetLastError());

    auto ref_cpu = ref.cpu();
    auto out_cpu = out.cpu();

    for (int i = 0; i < N; i++) {
      testing::RefTranspose(ref_cpu.data[i], in_cpu.data[i],
                            in_cpu.tensor_shape_span(i).data(), perm, D);
    }

    Check(out_cpu, ref_cpu);
  }
}


}  // namespace kernels
}  // namespace dali
