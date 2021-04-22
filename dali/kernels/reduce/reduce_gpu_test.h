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

#ifndef DALI_KERNELS_REDUCE_REDUCE_GPU_TEST_H_
#define DALI_KERNELS_REDUCE_REDUCE_GPU_TEST_H_

#include <gtest/gtest.h>
#include <random>
#include <utility>
#include "dali/core/tensor_view.h"
#include "dali/kernels/kernel_req.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/reduce/online_reducer.h"
#include "dali/kernels/reduce/reduce_test.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace kernels {
namespace testing {

template <typename Kernel, typename Out, typename In>
struct ReductionKernelTest {
  Kernel kernel;
  TestTensorList<In> in;
  TestTensorList<Out> out, ref;

  KernelContext ctx;
  ScratchpadAllocator sa;
  std::mt19937_64 rng{12345};


  template <typename... Args>
  KernelRequirements Setup(
      const TensorListShape<> &in_shape,
      const TensorListShape<> &ref_out_shape,
      span<const int> axes, bool keep_dims, bool batch,
      Args &&...args) {
    in.reshape(in_shape);
    ref.reshape(ref_out_shape);
    auto req = kernel.Setup(ctx, in_shape, axes, keep_dims, batch, std::forward<Args>(args)...);
    ASSERT_EQ(req.output_shapes.size(), 1), req;
    ASSERT_EQ(req.output_shapes[0], ref_out_shape), req;
    out.reshape(ref_out_shape);
    sa.Reserve(req.scratch_sizes);
    return req;
  }

  void FillData(In min_value, In max_value) {
    UniformRandomFill(in.cpu(), rng, 0, 255);
    CUDA_CALL(
      cudaMemsetAsync(out.gpu().data[0], -1, sizeof(Out) * out.gpu().num_elements(), stream()));
  }

  template <typename... Args>
  void Run(Args &&...args) {
    auto scratchpad = sa.GetScratchpad();
    ctx.scratchpad = &scratchpad;
    kernel.Run(ctx, out.gpu(stream()), in.gpu(stream()), std::forward<Args>(args)...);
  }

  template <typename... Args>
  void Check(Args &&...args) {
    auto out_cpu = out.cpu(stream());
    CUDA_CALL(cudaStreamSynchronize(stream()));
    dali::Check(out_cpu, ref.cpu(), std::forward<Args>(args)...);
  }

  cudaStream_t stream() const {
    return ctx.gpu.stream;
  }
};


}  // namespace testing
}  // namespace kernels
}  // namespace dali


#endif  // DALI_KERNELS_REDUCE_REDUCE_GPU_TEST_H_
