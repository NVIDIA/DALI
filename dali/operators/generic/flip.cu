// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime_api.h>
#include <vector>
#include "dali/kernels/imgproc/flip_gpu.cuh"
#include "dali/operators/generic/flip.h"
#include "dali/operators/generic/flip_util.h"
#include "dali/pipeline/data/views.h"

namespace dali {

template <>
Flip<GPUBackend>::Flip(const OpSpec &spec) : Operator<GPUBackend>(spec) {}

void RunKernel(TensorList<GPUBackend> &output, const TensorList<GPUBackend> &input,
               const std::vector<int32> &depthwise, const std::vector<int32> &horizontal,
               const std::vector<int32> &vertical, cudaStream_t stream) {
  DALI_TYPE_SWITCH(
      input.type().id(), DType,
      auto in_shape = TransformShapes(input.shape(), input.GetLayout());
      auto in_view = reshape<flip_ndim>(view<const DType>(input), in_shape);
      kernels::KernelContext ctx;
      ctx.gpu.stream = stream;
      kernels::FlipGPU<DType> kernel;
      auto reqs = kernel.Setup(ctx, in_view);
      auto out_shape = reqs.output_shapes[0].to_static<flip_ndim>();
      auto out_view = reshape<flip_ndim>(view<DType>(output), out_shape);
      kernel.Run(ctx, out_view, in_view, depthwise, vertical, horizontal);
  )
}

template <>
void Flip<GPUBackend>::RunImpl(Workspace<GPUBackend> &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  output.SetLayout(input.GetLayout());
  output.set_type(input.type());
  output.ResizeLike(input);
  auto curr_batch_size = ws.GetInputBatchSize(0);
  auto horizontal = GetHorizontal(ws, curr_batch_size);
  auto vertical = GetVertical(ws, curr_batch_size);
  auto depthwise = GetDepthwise(ws, curr_batch_size);
  RunKernel(output, input, depthwise, horizontal, vertical, ws.stream());
}

DALI_REGISTER_OPERATOR(Flip, Flip<GPUBackend>, GPU);

}  // namespace dali
