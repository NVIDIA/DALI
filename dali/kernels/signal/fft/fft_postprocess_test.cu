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

#include <gtest/gtest.h>
#include <random>
#include "dali/kernels/signal/fft/fft_postprocess.cuh"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/kernels/scratch.h"

namespace dali {
namespace kernels {
namespace signal {
namespace fft_postprocess {

TEST(FFTPostprocess, ToFreqMajor) {
  std::mt19937_64 rng;
  TensorListShape<2> in_shape, out_shape;
  std::uniform_int_distribution<int> dist(1, 500);
  int N = 10;
  int fft = 200;  // deliberately not a multiple of 32
  in_shape.resize(N);
  out_shape.resize(N);
  for (int i = 0; i < N; i++) {
    int len = dist(rng);
    in_shape.set_tensor_shape(i, { len, fft });
    out_shape.set_tensor_shape(i, { fft, len });
  }
  TestTensorList<float, 2> in, out, ref;
  in.reshape(in_shape);
  auto cpu_in = in.cpu();
  UniformRandomFill(cpu_in, rng, 0, 1);

  SpectrumToFreqMajor<float> tr;
  KernelContext ctx;
  ScratchpadAllocator sa;
  KernelRequirements req = tr.Setup(ctx, in.gpu());
  ASSERT_EQ(req.output_shapes.size(), 1u);
  ASSERT_EQ(req.output_shapes[0], out_shape);
  sa.Reserve(req.scratch_sizes);
  auto scratchpad = sa.GetScratchpad();
  ctx.scratchpad = &scratchpad;
  out.reshape(out_shape);
  tr.Run(ctx, out.gpu(), in.gpu());
  CUDA_CALL(cudaGetLastError());

  auto cpu_out = out.cpu();
  ref.reshape(out_shape);
  auto cpu_ref = ref.cpu();
  for (int i = 0; i < N; i++) {
    TensorView<StorageCPU, float, 2> in_tv = cpu_in[i];
    TensorView<StorageCPU, float, 2> ref_tv = cpu_ref[i];
    for (int y = 0; y < in_tv.shape[0]; y++)
      for (int x = 0; x < in_tv.shape[1]; x++)
        *ref_tv(x, y) = *in_tv(y, x);
  }

  Check(cpu_out, cpu_ref);
}

}  // fft_postprocess
}  // namespace signal
}  // namespace kernels
}  // namespace dali
