// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operators/geometric/flip.h"
#include <cuda_runtime_api.h>
#include <vector>
#include "dali/kernels/imgproc/flip_gpu.cuh"
#include "dali/pipeline/operators/geometric/flip_util.h"

namespace dali {

template <>
Flip<GPUBackend>::Flip(const OpSpec &spec) : Operator<GPUBackend>(spec) {}

void RunKernel(TensorList<GPUBackend> &output, const TensorList<GPUBackend> &input,
               const std::vector<int32> &horizontal,
               const std::vector<int32> &vertical, cudaStream_t stream) {
  DALI_TYPE_SWITCH(
      input.type().id(), DType,
      auto in_shape = TransformShapes(input.shape(), input.GetLayout() == DALI_NHWC);
      kernels::InListGPU<DType, 4> in_view(input.data<DType>(), in_shape);
      kernels::KernelContext ctx;
      ctx.gpu.stream = stream;
      kernels::FlipGPU<DType> kernel;
      auto reqs = kernel.Setup(ctx, in_view);
      kernels::OutListGPU<DType, 4> out_view(output.mutable_data<DType>(),
          reqs.output_shapes[0].to_static<4>());
      std::vector<int32> flip_z(horizontal.size());
      kernel.Run(ctx, out_view, in_view, flip_z, vertical, horizontal);
  )
}

template <>
void Flip<GPUBackend>::RunImpl(Workspace<GPUBackend> *ws, const int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);
  auto &output = ws->Output<GPUBackend>(idx);
  DALI_ENFORCE(input.GetLayout() == DALI_NHWC || input.GetLayout() == DALI_NCHW);
  output.SetLayout(input.GetLayout());
  output.set_type(input.type());
  output.ResizeLike(input);
  auto horizontal = GetHorizontal(ws);
  auto vertical = GetVertical(ws);
  RunKernel(output, input, horizontal, vertical, ws->stream());
}

DALI_REGISTER_OPERATOR(Flip, Flip<GPUBackend>, GPU);

}  // namespace dali
